from __future__ import print_function
import six
import os
import sys
import re
import string
import numpy as np
from collections import OrderedDict


oneapi_padding1D_map_to_cpp = {
    "valid": "{0}",
    "same": "{0}",
    "casual": "{1}"
}

oneapi_padding2D_map_to_cpp = {
    "valid": "{0, 0}",
    "same": "{0, 0}",
    "casual": "{1, 1}"
}

class Quantizer(object):
    def __init__(self, bits, hls_type):
        self.bits = bits
        self.hls_type = hls_type
    
    def __call__(self, data):
        raise NotImplementedError

class IntegerPrecisionType(object):
    def __init__(self, width=16, signed=True):
        self.width = width
        self.signed = signed
    
    def __str__(self):
        typestring = 'ap_{signed}int<{width}>'.format(signed='u' if not self.signed else '', width=self.width)
        return typestring

class FixedPrecisionType(object):
    def __init__(self, width=16, integer=6, signed=True, rounding_mode=None, saturation_mode=None, saturation_bits=None):
        self.width = width
        self.integer = integer
        self.signed = signed
        self.rounding_mode = rounding_mode
        self.saturation_mode = saturation_mode
        self.saturation_bits = saturation_bits
    
    def __str__(self):
        args = [self.width, self.integer, self.rounding_mode, self.saturation_mode, self.saturation_bits]
        args = ','.join([str(arg) for arg in args if arg is not None])
        typestring = 'ap_{signed}fixed<{args}>'.format(signed='u' if not self.signed else '', args=args)
        return typestring

def find_minimum_width(data, signed=True):
    """
    Helper function to find the minimum integer width to express all entries in the data array
    without saturation / overflow
    """
    maxdata = np.amax(np.abs(data))
    if maxdata == 0.:
        # fringe case (amax(abs(data)) == 0 -> data is uniformly zero)
        return 1

    log2max = np.log2(maxdata)

    iwidth = max(0, int(np.ceil(log2max)))
    if iwidth == int(np.floor(log2max)): # is a power-of-two integer -> need one extra bit
        iwidth += 1

    if signed:
        # add the sign bit
        iwidth += 1

    return iwidth

class HLSType(object):
    def __init__(self, name, precision, **kwargs):
        self.name = name.format(**kwargs)
        self.precision = precision

    def definition_cpp(self):
        return 'typedef {precision} {name};\n'.format(name=self.name, precision=self.precision)

class CompressedType(HLSType):
    def __init__(self, name, precision, index_precision, **kwargs):
        super(CompressedType, self).__init__('compressed_type{index}', precision, **kwargs)
        self.index_precision = index_precision

    def definition_cpp(self):
        cpp_fmt = ('typedef struct {name} {{ '
               '{index} row_index; '
               '{index} col_index; '
               '{precision} weight; }} {name};\n')
        return cpp_fmt.format(name=self.name, index=self.index_precision, precision=self.precision)

class Variable(object):
    def __init__(self, var_name, type_name, precision, **kwargs):
        self.name = var_name.format(**kwargs)
        self.type = HLSType(type_name, precision, **kwargs)
        self.cppname = re.sub(r'\W|^(?=\d)','_', self.name)

class ArrayVariable(Variable):
    def __init__(self, shape, dim_names, var_name='layer{index}', type_name='layer{index}_t', precision=None, pragma='partition', **kwargs):
        super(ArrayVariable, self).__init__(var_name, type_name, precision, **kwargs)
        self.shape = shape
        self.dim_names = dim_names
        self.pragma = pragma

    def get_shape(self):
        return zip(self.dim_names, self.shape)

    def definition_cpp(self):
        array_shape = self.size_cpp()
        return '{type} {name}[{shape}]'.format(type=self.type.name, name=self.cppname, shape=array_shape)

    def size(self):
        nelem = 1
        for dim in self.shape:
            nelem *= dim
        return nelem

    def size_cpp(self):
        return '*'.join([str(k) for k in self.dim_names])

class InplaceVariable():
    def __init__(self, shape, dim_names, proxy, **kwargs):
        self.shape = shape
        self.dim_names = dim_names
        self.type = proxy.type
        self.name = proxy.name
        self.size = proxy.size

    def get_shape(self):
        return zip(self.dim_names, self.shape)

    def definition_cpp(self):
        return None

    def size_cpp(self):
        return '*'.join([str(k) for k in self.dim_names])

class WeightVariable(Variable):
    def __init__(self, var_name, type_name, precision, data, quantizer=None, **kwargs):
        super(WeightVariable, self).__init__(var_name, type_name, precision, **kwargs)
        self.data = data
        self.nzeros = -1
        self.shape = list(self.data.shape)
        self.data_length = np.prod(self.data.shape)
        self.nonzeros = np.count_nonzero(self.data)
        self.nzeros = self.data_length - self.nonzeros
        self.min = np.min(self.data)
        self.max = np.max(self.data)
        self._iterator = None
        self.update_precision(precision)
        self.quantizer = quantizer

    def __iter__(self):
        self._iterator = np.nditer(self.data, order='C')
        return self

    def __next__(self):
        if not self._iterator.finished:
            value = self._iterator[0]
            self._iterator.iternext()
            return self.precision_fmt % value
        else:
            raise StopIteration

    next = __next__

    def update_precision(self, new_precision):
        self.type.precision = new_precision
        precision_str = str(self.type.precision)
        if 'int' in precision_str:
            self.precision_fmt = '%d'
        else:
            match = re.search('.+<(.+?)>', precision_str)
            if match is not None:
                precision_bits = match.group(1).split(',')
                width_bits = int(precision_bits[0])
                integer_bits = int(precision_bits[1])
                fractional_bits = integer_bits - width_bits
                lsb = 2 ** fractional_bits
                if lsb < 1:
                    # Use str to represent the float with digits, get the length
                    # to right of decimal point
                    decimal_spaces = len(str(lsb).split('.')[1])
                else:
                    decimal_spaces = len(str(2**integer_bits)) 
                self.precision_fmt = '%.{}f'.format(decimal_spaces)
            else:
                self.precision_fmt = '%f'

    def definition_cpp(self):
        return '{type} {name}[{size}]'.format(type=self.type.name, name=self.cppname, size=self.data_length)

class CompressedWeightVariable(WeightVariable):
    def __init__(self, var_name, type_name, precision, data, reuse_factor, quantizer=None, **kwargs):
        super(CompressedWeightVariable, self).__init__(var_name, type_name, precision, data, quantizer=quantizer, **kwargs)
        self.extra_zeros = 0
        self.data_length = np.prod(data.shape) - self.nzeros
        while self.data_length % reuse_factor != 0:
            self.extra_zeros += 1
            self.data_length += 1
        self.nonzeros = np.prod(data.shape) - self.nzeros + self.extra_zeros

        # Compress the array
        weights = []
        extra_nzero_cnt = self.extra_zeros
        it = np.nditer(data, order='C', flags=['multi_index'])
        max_idx = 0
        while not it.finished:
            val = it[0]
            if not (val == 0 and extra_nzero_cnt < 1):
                if val == 0:
                    extra_nzero_cnt -= 1
                if it.multi_index[0] > max_idx:
                    max_idx = it.multi_index[0]
                if it.multi_index[1] > max_idx:
                    max_idx = it.multi_index[1]
                weights.append([it.multi_index[1], it.multi_index[0], val])
            it.iternext()
        weights.sort()

        index_precision = 32
        if max_idx > 0:
            index_precision = int(np.log2(max_idx) + 1)
        self.type = CompressedType(type_name, precision, IntegerPrecisionType(width=index_precision, signed=False), **kwargs)

        self.data = weights

    def __iter__(self):
        self._iterator = iter(self.data)
        return self

    def __next__(self):
        value = next(self._iterator)
        value_fmt = self.precision_fmt % value[2]
        return '{ %u, %u, %s }' % (value[1], value[0], value_fmt)

    next = __next__

class Layer(object):
    def __init__(self, model, name, attributes, inputs, outputs=None):
        self.model = model
        self.name = name
        self.index = model.next_layer()
        self.inputs = inputs
        self.outputs = outputs
        if self.outputs is None:
            self.outputs = [self.name]

        self.attributes = attributes
        self.memory_descriptor = False

        self._function_template = self.model.config.backend.get_function_template(self.__class__.__name__)
        self._config_template = self.model.config.backend.get_config_template(self.__class__.__name__)
        self.include_list = self.model.config.backend.get_include_list(self.__class__.__name__)
        self._memory_template = self.model.config.backend.get_config_template("Memory")
        self.weights = OrderedDict()
        self.variables = OrderedDict()
        self.precision = OrderedDict()
        accum_t = HLSType(*reversed(self.model.config.get_precision(self, 'accum')))
        self.precision[accum_t.name] = accum_t
        self.set_attr('accum_t', accum_t.precision)
        self.reuse_factor = self.model.config.get_reuse_factor(self)

        layer_config = self.model.config.get_layer_config(self)
        for config_key, config_value in layer_config.items():
            if config_key in self.attributes:
                print('WARNING: Config parameter "{}" overwrites an existing attribute in layer "{}" ({})'.format(config_key, self.name, self.__class__.__name__))
            self.attributes[config_key] = config_value

        self.initialize()

    def initialize(self):
        raise NotImplementedError

    def set_attr(self, key, value):
        self.attributes[key] = value

    def get_attr(self, key, default=None):
        return self.attributes.get(key, default)

    def get_input_node(self, input_name=None):
        if input_name is not None:
            return self.model.graph.get(input_name)
        else:
            return self.model.graph.get(self.inputs[0])

    def get_input_variable(self, input_name=None):
        if input_name is not None:
            return self.model.get_layer_output_variable(input_name)
        else:
            return self.model.get_layer_output_variable(self.inputs[0])

    def get_output_nodes(self, output_name=None):
        if output_name is None:
            output_name = self.outputs[0]
        return [node for node in self.model.graph.values() if node.inputs[0] == output_name]

    @staticmethod
    def get_input_node_with_mem_desc(node):
        input_node = node.get_input_node()
        if input_node is None:
            return node
        elif input_node.memory_descriptor:
            return input_node
        else:
            return node.get_input_node_with_mem_desc(input_node)

    def get_output_variable(self, output_name=None):
        if output_name is not None:
            return self.variables[output_name]
        else:
            return next(iter(self.variables.values()))

    def get_weights(self, var_name=None):
        if var_name:
            return self.weights[var_name]

        return self.weights.values()

    def get_weights_precision(self):
        return self.weights['weight'].type.precision

    def get_variables(self):
        return self.variables.values()

    def add_output_variable(self, shape, dim_names, out_name=None, var_name='layer{index}_out', type_name='layer{index}_t', precision=None, pragma='auto'):
        if out_name is None:
            out_name = self.outputs[0]

        if precision is None:
            precision, _ = self.model.config.get_precision(self, var='result')

        if pragma == 'auto':
            if self.model.config.get_config_value('IOType') == 'io_serial':
                pragma = 'stream'
            else:
                if self.name in self.model.inputs:
                    pragma = 'reshape'
                else:
                    pragma = 'partition'

        out = ArrayVariable(shape, dim_names, var_name=var_name, type_name=type_name, precision=precision, pragma=pragma, index=self.index)

        self.variables[out_name] = out
        self.model.register_output_variable(out_name, out)

        self.precision[out.type.name] = out.type

    def add_weights(self, quantizer=None, compression=False):
        data = self.model.get_weights_data(self.name, 'kernel')

        self.add_weights_variable(name='weight', var_name='w{index}', data=data, quantizer=quantizer, compression=compression)

    def add_bias(self, quantizer=None):
        data = self.model.get_weights_data(self.name, 'bias')
        precision = None
        type_name = None
        if data is None:
            data = np.zeros(self.get_output_variable().shape[-1])
            precision = IntegerPrecisionType(width=1, signed=False)
            type_name = 'bias{index}_t'
            quantizer = None # Don't quantize non-existant bias

        self.add_weights_variable(name='bias', var_name='b{index}', type_name=type_name, precision=precision, data=data, quantizer=quantizer)

    def add_weights_variable(self, name, var_name=None, type_name=None, precision=None, data=None, quantizer=None, compression=False):
        if var_name is None:
            var_name = name + '{index}'

        if precision is None:
            precision, _ = self.model.config.get_precision(self, var=name)
        elif type_name is None:
            # If precision is specified but no type name is given, assign a dedicated
            # type name made from variable name and layer index
            type_name = name + '{index}_t'

        if type_name is None:
            _, type_name = self.model.config.get_precision(self, var=name)

        if data is None:
            data = self.model.get_weights_data(self.name, name)
        elif isinstance(data, six.string_types):
            data = self.model.get_weights_data(self.name, data)

        data_unquantized = data
        if quantizer is not None:
            precision = quantizer.hls_type
            type_name = name + '{index}_t'
            data = quantizer(data)

        if compression:
            var = CompressedWeightVariable(var_name, type_name=type_name, precision=precision, quantizer=quantizer, data=data, reuse_factor=self.reuse_factor, index=self.index)
        else:
            var = WeightVariable(var_name, type_name=type_name, precision=precision, quantizer=quantizer, data=data, index=self.index)

            var.data_unquantized = data_unquantized
        self.weights[name] = var
        self.precision[var.type.name] = var.type

    def _default_function_params(self):
        params = {}
        params['config'] = 'config{}'.format(self.index)
        params['input_t'] = self.get_input_variable().type.name
        params['output_t'] = self.get_output_variable().type.name
        params['input'] = self.get_input_variable().name
        params['output'] = self.get_output_variable().name

        return params

    def _default_config_params(self):
        params = {}
        params.update(self.attributes)
        params['index'] = self.index
        params['iotype'] = self.model.config.get_config_value('IOType')
        params['reuse'] = self.reuse_factor

        # data types
        for weight_name, variable in self.weights.items():
            params[weight_name + '_t'] = variable.type.name

        return params

    def _default_weight_params(self):
        """Returns dict with default weight parameters for oneAPI project"""
        params = {}
        params["layer_name"] = self.name
        params["memory_object"] = "weights"
        params["memory_object_type"] = "auto"
        input_dims = self.get_input_variable().shape
        output_dims = self.get_output_variable().shape
        if self.get_attr('data_format') == 'channels_first':
            weight_dims = str(output_dims).replace('[','').replace(']','')
            weight_dims += ", " + str(input_dims).replace('[','').replace(']','')
        else:
            weight_dims = str(output_dims[0]) + ", " + str(input_dims[-1])
            if len(input_dims) > 1:
                weight_dims += ',' + ','.join(map(str, input_dims[:-1]))
        params["dims"] = weight_dims
        params["data_type"] = self.get_weights_precision()
        params["format_tag"] = string.ascii_lowercase[:len(input_dims)+len(output_dims)]

        return params

    def _conv_weight_params(self):
        """Returns dict with convolutional weight parameters for oneAPI project"""
        params = {}
        params["layer_name"] = self.name
        params["memory_object"] = "weights"
        params["placeholder"] = "placeholder_"
        params["memory_object_type"] = "auto"
        input_dims = self.get_input_variable().shape
        output_dims = self.get_output_variable().shape
        weight_dims = str(output_dims[-1]) + ", " + str(input_dims[-1])
        if self.get_attr('filt_height'):
            weight_dims += ", " + str(self.get_attr('filt_height'))
        weight_dims += ", " + str(self.get_attr('filt_width'))
        params["dims"] = weight_dims
        params["data_type"] = self.get_weights_precision()
        params["format_tag"] = string.ascii_lowercase[:len(input_dims) + 1]

        return params

    def _default_bias_params(self):
        """Returns dict with default bias parameters for oneAPI project"""
        params = {}
        params["layer_name"] = self.name
        params["memory_object"] = "bias"
        params["memory_object_type"] = "auto"
        params["placeholder"] = ""
        if self.get_attr('data_format') == 'channels_first':
            bias_dim = self.get_output_variable().shape[0]
        else:
            bias_dim = self.get_output_variable().shape[-1]
        params["dims"] = bias_dim
        params["data_type"] = self.get_weights_precision()
        params["format_tag"] = "x"

        return params

    def get_layer_precision(self):
        return self.precision

    # myproject.cpp/h
    def function_cpp(self):
        raise NotImplementedError

    # parameters.h
    def config_cpp(self):
        raise NotImplementedError

    def get_numbers_cpp(self):
        numbers = ''
        for k, v in self.get_output_variable().get_shape():
            numbers += '#define {} {}\n'.format(k,v)

        return numbers

    def precision_cpp(self):
        return 'typedef {precision} layer{index}_t;'.format(precision=self.get_output_variable().precision, index=self.index)

class Input(Layer):
    def initialize(self):
        shape = self.attributes['input_shape']
        if shape[0] is None:
            shape = shape[1:]
        dims = ['N_INPUT_{}_{}'.format(i, self.index) for i in range(1, len(shape) + 1)]
        if self.index == 1:
            default_type_name = 'input_t'
        else:
            default_type_name = 'input{}_t'.format(self.index)
        type_name = self.attributes.get('type_name', default_type_name)
        precision = self.attributes.get('precision', None)
        self.add_output_variable(shape, dims, var_name=self.name, type_name=type_name, precision=precision)

    def function_cpp(self):
        return None

    def config_cpp(self):
        return None
    
    def definition_dcpp(self):
        input_params = {}
        input_params["layer_name"] = "input"
        input_params["memory_object"] = "data"
        input_params["memory_object_type"] = ""
        input_params["placeholder"] = ""
        batch_size = f"{self.model.batch_size}, "
        input_dims = self.attributes['input_shape']
        if self.get_attr('data_format') == 'channels_first':
            input_params["dims"] = batch_size + str(input_dims).replace('[','').replace(']','')
        else:
            input_params["dims"] = batch_size + str(input_dims[-1])
            if len(input_dims) > 1:
                input_params["dims"] += ',' + ','.join(map(str, input_dims[:-1]))
        input_params["data_type"] = "f32"
        input_params["format_tag"] = string.ascii_lowercase[:len(input_dims)+1]

        return self._memory_template.format(**input_params)

class Reshape(Layer):
    def initialize(self):
        shape = self.attributes['target_shape']
        if shape[0] is None:
            shape = shape[1:]
        dims = ['N_SIZE_{}_{}'.format(i, self.index) for i in range(1, len(shape) + 1)]

        out_name = self.outputs[0]
        proxy = self.get_input_variable()
        out = InplaceVariable(shape, dims, proxy, index=self.get_input_node().index)

        self.variables[out_name] = out
        self.model.register_output_variable(out_name, out)

    def function_cpp(self):
        return None

    def config_cpp(self):
        return None

class Dense(Layer):
    def initialize(self):
        shape = [self.attributes['n_out']]
        dims = ['N_LAYER_{}'.format(self.index)]
        compression = self.model.config.get_compression(self)
        if self.model.config.is_resource_strategy(self):
            if self.model.config.backend.name == 'Vivado':
                self.model.config.backend.set_closest_reuse_factor(self)
            if compression:
                self.set_attr('strategy', 'compressed')
            else:
                self.set_attr('strategy', 'large')
        else:
            self.set_attr('strategy', 'latency')
        self.add_output_variable(shape, dims)
        self.add_weights(quantizer=self.get_attr('weight_quantizer'), compression=compression)
        index_t = IntegerPrecisionType(width=1, signed=False)
        if self.model.config.is_resource_strategy(self):
            if self.model.config.get_compression(self):
                index_t = self.get_weights('weight').type.index_precision
            elif self.model.config.backend.name == 'Vivado':
                self.weights['weight'].data = np.transpose(self.weights['weight'].data)
        if self.model.config.backend.name == "oneAPI":
            oneAPI_shape = self.get_input_variable().shape + [self.get_attr('n_out')]
            reshaped_weights = self.weights['weight'].data.reshape(oneAPI_shape) # there is no Flatten layer in hls4ml model and oneAPI expects different weights format
            if len(oneAPI_shape) > 3:
                self.weights['weight'].data = np.transpose(reshaped_weights, axes=[3, 2, 0, 1]) # (H * W * I, O) => (O, I, H, W)
            else:
                self.weights['weight'].data = np.transpose(reshaped_weights)
        self.set_attr('index_t', index_t)
        self.add_bias(quantizer=self.get_attr('bias_quantizer'))
        self.memory_descriptor = True

    def function_cpp(self):
        params = self._default_function_params()
        params['strategy'] = self.get_attr('strategy')
        params['w'] = self.get_weights('weight').name
        params['b'] = self.get_weights('bias').name

        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()
        params['n_in'] = self.get_input_variable().size_cpp()
        params['n_out'] = self.get_output_variable().size_cpp()
        params['nzeros'] = self.get_weights('weight').nzeros
        params['nonzeros'] = self.get_weights('weight').nonzeros

        return self._config_template.format(**params)

    def definition_dcpp(self):
        """Returns oneAPI definition for Dense Layer"""
        dense_params = {}
        dense_params["layer_name"] = self.name
        dense_params["memory_object_type"] = "" if "output" in self.name else "auto"
        dense_params["data_type"] = self.get_weights_precision()
        batch_size = f"{self.model.batch_size}, "
        output_dims = self.get_output_variable().shape
        dense_params["output_dims"] = batch_size + str(output_dims).replace('[','').replace(']','')
        if self.get_input_node().index == 1:
            dense_params["input_desc"] = "input_data_md"
            dense_params["input_memory"] = "input_data_memory"
        else:
            input_layer = self.get_input_node_with_mem_desc(self)
            dense_params["input_desc"] = f"{input_layer.name}_memory.get_desc()"
            dense_params["input_memory"] = f"{input_layer.name}_memory"
        dense_config = self._config_template.format(**dense_params)

        weight_params = self._default_weight_params()
        weight_params['placeholder'] = "placeholder_"
        weight_config = self._memory_template.format(**weight_params)

        bias_params = self._default_bias_params()
        bias_config = self._memory_template.format(**bias_params)

        return weight_config + bias_config + dense_config

class Conv1D(Layer):
    def initialize(self):
        if self.get_attr('data_format') == 'channels_last':
            shape = [self.attributes['n_out'], self.attributes['n_filt']]
            dims = ['N_OUTPUTS_{}'.format(self.index), 'N_FILT_{}'.format(self.index)]
        else:
            shape = [self.attributes['n_filt'], self.attributes['n_out']]
            dims = ['N_FILT_{}'.format(self.index), 'N_OUTPUTS_{}'.format(self.index)]

        self.add_output_variable(shape, dims)
        self.add_weights(quantizer = self.get_attr('weight_quantizer'))
        self.add_bias(quantizer = self.get_attr('bias_quantizer'))
        if self.model.config.is_resource_strategy(self):
            self.set_attr('strategy', 'large')
            if self.model.config.backend.name == 'Vivado':
                self.model.config.backend.set_closest_reuse_factor(self)
                self.weights['weight'].data = np.transpose(self.weights['weight'].data, axes=[2, 1, 0]) #(W,C,F) => (F,C,W)
        else:
            self.set_attr('strategy', 'latency')
        if self.model.config.backend.name == "oneAPI":
            self.weights['weight'].data = np.transpose(self.weights['weight'].data, axes=[2, 1, 0]) #(W,I,O) => (O,I,W)
        self.memory_descriptor = True

    def function_cpp(self):
        params = self._default_function_params()
        params['strategy'] = self.get_attr('strategy')
        params['data_format'] = 'cf' if self.get_attr('data_format') == 'channels_first' else 'cl'
        params['w'] = self.get_weights('weight').name
        params['b'] = self.get_weights('bias').name

        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()
        input_dims = self.get_input_variable().dim_names
        if self.get_attr('data_format') == 'channels_last':
            params['n_in'] = '*'.join([str(k) for k in input_dims[:-1]])
            params['n_chan'] = input_dims[-1]
        else:
            params['n_in'] = '*'.join([str(k) for k in input_dims[1:]])
            params['n_chan'] = input_dims[0]
        params['dilation'] = self.get_attr('dilation', 1)
        params['n_filt'] = 'N_FILT_{}'.format(self.index)
        params['n_out'] = 'N_OUTPUTS_{}'.format(self.index)
        params['nzeros'] = self.get_weights('weight').nzeros
        params['config_t'] = 'std::nullptr_t'

        if self.model.config.is_resource_strategy(self):
            params['config_t'] = 'config{}_mult'.format(self.index)
            conv_config = self._config_template[0].format(**params)

            mult_params = self._default_config_params()
            mult_params['n_in'] = self.get_attr('n_chan') * self.get_attr('filt_width')
            mult_params['n_out'] = self.get_attr('n_filt')
            mult_config = self._config_template[1].format(**mult_params)

            return mult_config + '\n' + conv_config
        else:
            return self._config_template[0].format(**params)
    
    def definition_dcpp(self):
        """Returns oneAPI definition for Conv1D Layer"""
        conv1d_params = {}
        conv1d_params["layer_name"] = self.name
        conv1d_params["memory_object_type"] = "" if "output" in self.name else "auto"
        conv1d_params["data_type"] = self.get_weights_precision()
        batch_size = f"{self.model.batch_size}, "
        output_dims = self.get_output_variable().shape
        conv1d_params["output_dims"] = batch_size + str(output_dims[-1]) + ', ' + str(output_dims[0])
        if self.get_input_node().index == 1:
            conv1d_params["input_desc"] = "input_data_md"
            conv1d_params["input_memory"] = "input_data_memory"
        else:
            input_layer = self.get_input_node_with_mem_desc(self)
            conv1d_params["input_desc"] = f"{input_layer.name}_memory.get_desc()"
            conv1d_params["input_memory"] = f"{input_layer.name}_memory"
        conv1d_params["strides"] = '{' + str(self.get_attr('strides', "1")) + '}'
        conv1d_params["padding"] = oneapi_padding1D_map_to_cpp[self.get_attr('padding', 'valid')]
        conv1d_config = self._config_template.format(**conv1d_params)

        weight_params = self._conv_weight_params()
        weight_config = self._memory_template.format(**weight_params)

        bias_params = self._default_bias_params()
        bias_config = self._memory_template.format(**bias_params)

        return weight_config + bias_config + conv1d_config

class Conv2D(Layer):
    def initialize(self):
        if self.get_attr('data_format') == 'channels_last':
            shape = [self.attributes['out_height'], self.attributes['out_width'], self.attributes['n_filt']]
            dims = ['OUT_HEIGHT_{}'.format(self.index), 'OUT_WIDTH_{}'.format(self.index), 'N_FILT_{}'.format(self.index)]
        else:
            shape = [self.attributes['n_filt'], self.attributes['out_height'], self.attributes['out_width']]
            dims = ['N_FILT_{}'.format(self.index), 'OUT_HEIGHT_{}'.format(self.index), 'OUT_WIDTH_{}'.format(self.index)]
        self.add_output_variable(shape, dims)
        self.add_weights(quantizer=self.get_attr('weight_quantizer'))
        self.add_bias(quantizer=self.get_attr('bias_quantizer'))
        if self.model.config.is_resource_strategy(self):
            self.set_attr('strategy', 'large')
            if self.model.config.backend.name == 'Vivado':
                self.model.config.backend.set_closest_reuse_factor(self)
                self.weights['weight'].data = np.transpose(self.weights['weight'].data, axes=[3, 2, 0, 1]) #(H,W,C,F) => (F,C,H,W)
        else:
            self.set_attr('strategy', 'latency')
        if self.model.config.backend.name == "oneAPI":
            self.weights['weight'].data = np.transpose(self.weights['weight'].data, axes=[3, 2, 0, 1]) # (H,W,I,O) => (O,I,H,W)
        self.memory_descriptor = True

    def function_cpp(self):
        params = self._default_function_params()
        params['strategy'] = self.get_attr('strategy')
        params['data_format'] = 'cf' if self.get_attr('data_format') == 'channels_first' else 'cl'
        params['w'] = self.get_weights('weight').name
        params['b'] = self.get_weights('bias').name

        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()
        if self.get_attr('data_format') == 'channels_last':
            params['in_height'] = self.get_input_variable().dim_names[0]
            params['in_width'] = self.get_input_variable().dim_names[1]
            params['n_chan'] = self.get_input_variable().dim_names[2]
            params['out_height'] = self.get_output_variable().dim_names[0]
            params['out_width'] = self.get_output_variable().dim_names[1]
            params['n_filt'] = self.get_output_variable().dim_names[2]
        else:
            params['n_chan'] = self.get_input_variable().dim_names[0]
            params['in_height'] = self.get_input_variable().dim_names[1]
            params['in_width'] = self.get_input_variable().dim_names[2]
            params['n_filt'] = self.get_output_variable().dim_names[0]
            params['out_height'] = self.get_output_variable().dim_names[1]
            params['out_width'] = self.get_output_variable().dim_names[2]
        params['dilation'] = self.get_attr('dilation', 1)
        params['nzeros'] = self.get_weights('weight').nzeros
        params['config_t'] = 'std::nullptr_t'

        if self.model.config.is_resource_strategy(self):
            params['config_t'] = 'config{}_mult'.format(self.index)
            conv_config = self._config_template[0].format(**params)

            mult_params = self._default_config_params()
            mult_params['n_in'] = self.get_attr('n_chan') * self.get_attr('filt_height') * self.get_attr('filt_width')
            mult_params['n_out'] = self.get_attr('n_filt')
            mult_config = self._config_template[1].format(**mult_params)

            return mult_config + '\n' + conv_config
        else:
            return self._config_template[0].format(**params)

    def definition_dcpp(self):
        """Returns oneAPI definition for Conv2D Layer"""
        conv2d_params = {}
        conv2d_params["layer_name"] = self.name
        conv2d_params["memory_object_type"] = "" if "output" in self.name else "auto"
        conv2d_params["data_type"] = self.get_weights_precision()
        batch_size = f"{self.model.batch_size}, "
        output_dims = self.get_output_variable().shape
        conv2d_params["output_dims"] = batch_size + str(output_dims[-1]) + ', ' + str(output_dims[0]) + ', ' + str(output_dims[1])
        if self.get_input_node().index == 1:
            conv2d_params["input_desc"] = "input_data_md"
            conv2d_params["input_memory"] = "input_data_memory"
        else:
            input_layer = self.get_input_node_with_mem_desc(self)
            conv2d_params["input_desc"] = f"{input_layer.name}_memory.get_desc()"
            conv2d_params["input_memory"] = f"{input_layer.name}_memory"
        conv2d_params["strides"] = self.get_attr('stride', '{1, 1}')
        conv2d_params["padding"] = oneapi_padding2D_map_to_cpp[self.get_attr('padding', 'valid')]
        conv2d_config = self._config_template.format(**conv2d_params)

        weight_params = self._conv_weight_params()
        weight_config = self._memory_template.format(**weight_params)

        bias_params = self._default_bias_params()
        bias_config = self._memory_template.format(**bias_params)

        return weight_config + bias_config + conv2d_config

class Pooling1D(Layer):
    def initialize(self):
        shape = [self.attributes['n_out'], self.attributes['n_filt']]
        dims = ['N_OUTPUTS_{}'.format(self.index), 'N_FILT_{}'.format(self.index)]
        self.add_output_variable(shape, dims)
        self.set_attr('pool_op', self.get_attr('class_name').split('Pooling')[0])
        self.memory_descriptor = True

    def function_cpp(self):
        params = self._default_function_params()

        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()
        params['n_in'] = self.get_input_variable().size_cpp()
        params['n_out'] = self.get_output_variable().size_cpp()

        return self._config_template.format(**params)

    def definition_dcpp(self):
        """Returns oneAPI definition for Pooling1D"""
        pool1d_params = {}
        pool1d_params["layer_name"] = self.name
        input_layer = self.get_input_node_with_mem_desc(self)
        pool1d_params["input_desc"] = f"{input_layer.name}_memory.get_desc()"
        pool1d_params["memory_object_type"] = "" if "output" in self.name else "auto"
        pool1d_params["data_type"] = input_layer.get_weights_precision()
        batch_size = f"{self.model.batch_size}, "
        output_dims = self.get_output_variable().shape
        pool1d_params["output_dims"] = batch_size + str(output_dims[-1]) + ', ' + str(output_dims[0])
        pool1d_params["strides"] = '{' + str(self.get_attr('stride', '2')) + '}'
        pool1d_params["padding"] = oneapi_padding1D_map_to_cpp[self.get_attr('padding', 'valid')]
        pool1d_params["kernel"] = str(self.attributes['n_out'])
        if self.get_input_node().index == 1:
            pool1d_params["input_desc"] = "input_data_md"
            pool1d_params["input_memory"] = "input_data_memory"
        else:
            input_layer = self.get_input_node_with_mem_desc(self)
            pool1d_params["input_desc"] = f"{input_layer.name}_memory.get_desc()"
            pool1d_params["input_memory"] = f"{input_layer.name}_memory"
        pool1d_config = self._config_template.format(**pool1d_params)

        return pool1d_config

class Pooling2D(Layer):
    def initialize(self):
        shape = [self.attributes['out_height'], self.attributes['out_width'], self.attributes['n_filt']]
        dims = ['OUT_HEIGHT_{}'.format(self.index), 'OUT_WIDTH_{}'.format(self.index), 'N_FILT_{}'.format(self.index)]
        self.add_output_variable(shape, dims)
        self.set_attr('pool_op', self.get_attr('class_name').split('Pooling')[0])
        self.memory_descriptor = True

    def function_cpp(self):
        params = self._default_function_params()
        params['data_format'] = 'cf' if self.get_attr('data_format') == 'channels_first' else 'cl'
        
        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()
        params['n_in'] = self.get_input_variable().dim_names[0]
        params['in_width'] = self.get_input_variable().dim_names[1]
        params['out_height'] = self.get_output_variable().dim_names[0]
        params['out_width'] = self.get_output_variable().dim_names[1]
        params['n_filt'] = self.get_output_variable().dim_names[2]

        return self._config_template.format(**params)

    def definition_dcpp(self):
        """Returns oneAPI definition for Pooling2D"""
        pool2d_params = {}
        pool2d_params["layer_name"] = self.name
        input_layer = self.get_input_node_with_mem_desc(self)
        pool2d_params["input_desc"] = f"{input_layer.name}_memory.get_desc()"
        pool2d_params["memory_object_type"] = "" if "output" in self.name else "auto"
        pool2d_params["data_type"] = input_layer.get_weights_precision()
        batch_size = f"{self.model.batch_size}, "
        output_dims = self.get_output_variable().shape
        pool2d_params["output_dims"] = batch_size + str(output_dims[-1]) + ', ' + str(output_dims[0]) + ', ' + str(output_dims[1])
        pool2d_params["strides"] = self.get_attr('stride', '{2, 2}')
        pool2d_params["padding"] = oneapi_padding2D_map_to_cpp[self.get_attr('padding', 'valid')]
        pool2d_params["kernel"] = str(self.attributes['pool_height']) + ', ' +  str(self.attributes['pool_width'])
        if self.get_input_node().index == 1:
            pool2d_params["input_desc"] = "input_data_md"
            pool2d_params["input_memory"] = "input_data_memory"
        else:
            input_layer = self.get_input_node_with_mem_desc(self)
            pool2d_params["input_desc"] = f"{input_layer.name}_memory.get_desc()"
            pool2d_params["input_memory"] = f"{input_layer.name}_memory"
        pool2d_config = self._config_template.format(**pool2d_params)

        return pool2d_config


class Activation(Layer):
    def initialize(self):
        inp = self.get_input_variable()
        shape = inp.shape
        dims = inp.dim_names
        self.add_output_variable(shape, dims)
        if self.model.config.backend.name == 'Vivado':
            if 'table_t' not in self.attributes:
                self.set_attr('table_t', FixedPrecisionType(width=18, integer=8))
            if 'table_size' not in self.attributes:
                self.set_attr('table_size', 1024)

    def function_cpp(self):
        params = self._default_function_params()
        params['activation'] = self.get_attr('activation').lower()
        params['config'] = '{}_config{}'.format(self.get_attr('activation'), self.index)

        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()
        params['type'] = self.get_attr('activation')
        params['n_in'] = self.get_input_variable().size_cpp()

        return self._config_template.format(**params)
    
    def definition_dcpp(self):
        """Returns oneAPI definition for Activation Function"""
        params = {}
        params["type"] = self.get_attr('activation').lower()
        params["layer_name"] = self.name
        input_layer = self.get_input_node_with_mem_desc(self)
        params["input_desc"] = f"{input_layer.name}_memory.get_desc()"
        params["alpha"] = self.get_attr('activ_param', 0.0)
        params["input_memory"] = f"{input_layer.name}_memory"
        params["output_memory"] = f"{input_layer.name}_memory"

        return self._config_template.format(**params)

class ParametrizedActivation(Activation):
    def function_cpp(self):
        params = self._default_function_params()
        params['activation'] = self._get_act_function_name()
        params['param'] = self.get_attr('activ_param', 1.0)
        params['config'] = '{}_config{}'.format(self.get_attr('activation'), self.index)

        return [self._function_template.format(**params)]

    def _get_act_function_name(self):
        act = self.get_attr('activation').lower()
        if act == 'leakyrelu':
            return 'leaky_relu'
        elif act == 'thresholdedrelu':
            return 'thresholded_relu'
        else:
            return act # ELU activation

class PReLU(Activation):
    def initialize(self):
        super(PReLU, self).initialize()
        self.add_weights_variable(name='alpha', var_name='a{index}')

    def function_cpp(self):
        params = self._default_function_params()
        params['activation'] = self.get_attr('activation').lower()
        params['param'] = self.get_weights('alpha').name
        params['config'] = '{}_config{}'.format(self.get_attr('activation'), self.index)

        return [self._function_template.format(**params)]

class Softmax(Activation):
    def initialize(self):
        super(Softmax, self).initialize()
        if self.model.config.backend.name == 'Vivado':
            if 'exp_table_t' not in self.attributes:
                self.set_attr('exp_table_t', self.get_attr('table_t'))
            if 'inv_table_t' not in self.attributes:
                self.set_attr('inv_table_t', self.get_attr('table_t'))
            if self.model.config.is_resource_strategy(self):
                # 'resource' strategy = 'latency' for Softmax
                self.set_attr('implementation', 'latency')
            else:
                self.set_attr('implementation', self.model.config.get_strategy(self).lower())

    def definition_dcpp(self):
        """Returns oneAPI definition for Activation Function"""
        params = {}
        params["layer_name"] = self.name
        input_layer = self.get_input_node_with_mem_desc(self)
        params["input_desc"] = f"{input_layer.name}_memory.get_desc()"
        params["axis"] = self.get_attr('axis', 1)
        params["input_memory"] = f"{input_layer.name}_memory"
        params["output_memory"] = f"{input_layer.name}_memory"

        return self._config_template.format(**params)
        
class BatchNormalization(Layer):
    def initialize(self):
        inp = self.get_input_variable()
        shape = inp.shape
        dims = inp.dim_names
        self.add_output_variable(shape, dims)

        gamma = self.model.get_weights_data(self.name, 'gamma')
        beta = self.model.get_weights_data(self.name, 'beta')
        mean = self.model.get_weights_data(self.name, 'moving_mean')
        var = self.model.get_weights_data(self.name, 'moving_variance')

        scale = gamma / np.sqrt(var + self.get_attr('epsilon'))
        bias = beta - gamma * mean / np.sqrt(var + self.get_attr('epsilon'))

        self.add_weights_variable(name='scale', var_name='s{index}', data=scale)
        self.add_weights_variable(name='bias', var_name='b{index}', data=bias)

    def function_cpp(self):
        params = self._default_function_params()
        params['scale'] = self.get_weights('scale').name
        params['bias'] = self.get_weights('bias').name

        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()
        params['n_in'] = self.get_input_variable().size_cpp()

        return self._config_template.format(**params)

class Merge(Layer):
    def initialize(self):
        assert(len(self.inputs) == 2)
        inp1 = self.get_input_variable(self.inputs[0])
        inp2 = self.get_input_variable(self.inputs[1])
        shape = inp1.shape
        assert(inp1.shape == inp2.shape)
        dims = inp1.dim_names
        self.add_output_variable(shape, dims)

    def function_cpp(self):
        params = {}
        params['merge'] = self.get_attr('op').lower()
        params['config'] = 'config{}'.format(self.index)
        params['input1_t'] = self.get_input_variable(self.inputs[0]).type.name
        params['input2_t'] = self.get_input_variable(self.inputs[1]).type.name
        params['output_t'] = self.get_output_variable().type.name
        params['input1'] = self.get_input_variable(self.inputs[0]).name
        params['input2'] = self.get_input_variable(self.inputs[1]).name
        params['output'] = self.get_output_variable().name

        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()
        params['n_elem'] = self.get_input_variable(self.inputs[0]).size_cpp()

        return self._config_template.format(**params)

class Concatenate(Merge):
    def initialize(self):
        assert(len(self.inputs) == 2)
        inp1 = self.get_input_variable(self.inputs[0])
        inp2 = self.get_input_variable(self.inputs[1])
        axis = self.attributes['axis']
        shape = inp1.shape[:]
        shape[axis] += inp2.shape[axis]
        rank = len(shape)
        if rank > 1:
            dims = ['OUT_CONCAT_{}_{}'.format(i, self.index) for i in range(rank)]
        else:
            dims = ['OUT_CONCAT_{}'.format(self.index)]
        self.add_output_variable(shape, dims)

    def config_cpp(self):
        params = self._default_config_params()
        for i in range(3):
            params.setdefault('n_elem1_{}'.format(i), 0)
            params.setdefault('n_elem2_{}'.format(i), 0)
        inp1 = self.get_input_variable(self.inputs[0])
        inp2 = self.get_input_variable(self.inputs[1])
        for i, (s1, s2) in enumerate(zip(inp1.shape, inp2.shape)):
            params['n_elem1_{}'.format(i)] = s1
            params['n_elem2_{}'.format(i)] = s2

        return self._config_template.format(**params)

class BiasAdd(Merge): # TensorFlow's operator that gets merged into Dense/Conv
    def initialize(self):
        inp = self.get_input_variable(self.inputs[0])
        shape = inp.shape
        dims = inp.dim_names
        self.add_bias()
        self.add_output_variable(shape, dims)

    def function_cpp(self):
        raise Exception('Layer {} should not be exported to HLS'.format(self.__class__.__name__))

    def config_cpp(self):
        raise Exception('Layer {} should not be exported to HLS'.format(self.__class__.__name__))

class Resize(Layer):
    def initialize(self):
        shape = [self.get_attr('new_height'), self.get_attr('new_width'), self.get_attr('n_chan')]
        dims = ['OUT_HEIGHT_{}'.format(self.index), 'OUT_WIDTH_{}'.format(self.index), 'N_CHAN_{}'.format(self.index)]
        self.add_output_variable(shape, dims)

    def function_cpp(self):
        params = self._default_function_params()
        params['algorithm'] = self.get_attr('algorithm')

        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()

        return self._config_template.format(**params)

class Transpose(Layer):
    def initialize(self):
        inp = self.get_input_variable(self.inputs[0])
        perm = self.get_attr('perm')
        self.set_attr('dim', '{}d'.format(len(inp.shape)))
        if len(perm) == 4 and perm[0] == 0:
            perm = [i - 1 for i in perm[1:]]
        shape = [inp.shape[i] for i in perm]
        self.set_attr('perm_str', ','.join([str(i) for i in perm]))
        if len(shape) == 2:
            dims = ['OUT_HEIGHT_{}'.format(self.index), 'OUT_WIDTH_{}'.format(self.index)]
            self.set_attr('depth', 1)
            self.set_attr('height', shape[0])
            self.set_attr('width', shape[1])
        else:
            dims = ['OUT_DEPTH_{}'.format(self.index), 'OUT_HEIGHT_{}'.format(self.index), 'OUT_WIDTH_{}'.format(self.index)]
            self.set_attr('depth', shape[0])
            self.set_attr('height', shape[1])
            self.set_attr('width', shape[2])
        self.add_output_variable(shape, dims)

    def function_cpp(self):
        params = self._default_function_params()
        params['dim'] = self.get_attr('dim')

        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()

        return self._config_template.format(**params)

class GarNet(Layer):
    ref_impl = False

    def initialize(self):
        reuse_factor = self.model.config.get_reuse_factor(self)
        if self.attributes['n_vertices'] % reuse_factor != 0:
            raise Exception('GarNet vertex loop has no bound check; number of vertices must be divisible by the reuse factor ({}).'.format(reuse_factor))

        self._initialize_transforms()

        # A bit controvertial but we are going to reshape the input variable here
        input_array = self.get_input_variable(self.inputs[0])
        partition_factor = input_array.shape[1] * (input_array.shape[0] // reuse_factor)
        input_array.pragma = ('partition', 'cyclic', partition_factor)
        
        if self.attributes['collapse']:
            shape = [self._output_features]
            dims = ['OUT_FEATURES_{}'.format(self.index)]
            pragma = 'partition'
        else:
            shape = [self.attributes['n_vertices'], self._output_features]
            dims = ['VERTICES_{}'.format(self.index),'OUT_FEATURES_{}'.format(self.index)]
            partition_factor = self._output_features * (self.attributes['n_vertices'] // reuse_factor)
            pragma = ('partition', 'cyclic' , partition_factor)

        self.add_output_variable(shape, dims, pragma=pragma)

    def _initialize_transforms(self):
        n_propagate = self.attributes['n_propagate']
        n_aggregators = self.attributes['n_aggregators']
        n_out_features = self.attributes['n_out_features']

        if self.ref_impl:
            weights_source = [
                ('input_transform', 'FLR', 'kernel'),
                ('input_transform', 'FLR', 'bias'),
                ('aggregator_distance', 'S', 'kernel'),
                ('aggregator_distance', 'S', 'bias'),
                ('output_transform', 'Fout', 'kernel'),
                ('output_transform', 'Fout', 'bias')
            ]

        else:
            quantize = (self.get_attr('quantizer') is not None)
            kernel, bias = self._make_input_transform_weights(n_propagate, n_aggregators, n_out_features, quantize=quantize)

            self._add_variable('input_transform_weights', 'input_transform_w{index}', kernel, frac_width=10, quantize=quantize)
            self._add_variable('input_transform_biases', 'input_transform_b{index}', bias, frac_width=10, quantize=quantize)
            #dummy
            self.add_weights_variable(name='output_transform_weights', var_name='output_transform_w{index}', data=np.ones(1))

            weights_source = [
                ('aggregator_distance', 'S', 'kernel'),
                ('aggregator_distance', 'S', 'bias'),
                ('output_transform', 'Fout', 'bias')
            ]

        for op_name, lname, wtype in weights_source:
            data = self.model.get_weights_data(self.name, '{name}/{lname}_{wtype}:0'.format(name=self.name, lname=lname, wtype=wtype))
            if wtype == 'kernel':
                data = data.transpose((1, 0))
                vtype = 'weights'
            else:
                vtype = 'biases'

            name = '{}_{}'.format(op_name, vtype)
            var_name = '{}_{}{{index}}'.format(op_name, vtype[0])

            self._add_variable(name, var_name, data, frac_width=10, quantize=False)

        self._output_features = self.attributes['n_out_features']

    def _make_input_transform_weights(self, n_propagate, n_aggregators, n_out_features, quantize=False, sublayer=''):
        # Due to linearity of the input transform, input weights and biases can be contracted away at conversion time

        output_transform_kernel = self.model.get_weights_data(self.name, '{name}/Fout{sublayer}_kernel:0'.format(name=self.name, sublayer=sublayer)) # [(n_aggregators, n_propagate), n_out_features]
        output_transform_kernel = output_transform_kernel.reshape((n_aggregators, n_propagate, n_out_features))
        if quantize:
            output_transform_kernel = self.get_attr('quantizer')(output_transform_kernel)

        input_transform_kernel = self.model.get_weights_data(self.name, '{name}/FLR{sublayer}_kernel:0'.format(name=self.name, sublayer=sublayer)) # [n_in_features, n_propagate]
        if quantize:
            input_transform_kernel = self.get_attr('quantizer')(input_transform_kernel)
        data = np.dot(input_transform_kernel, output_transform_kernel) # [n_in_features, n_aggregators, n_out_features]
        kernel = data.transpose((2, 1, 0))

        input_transform_bias = self.model.get_weights_data(self.name, '{name}/FLR{sublayer}_bias:0'.format(name=self.name, sublayer=sublayer)) # [n_propagate]
        if quantize:
            input_transform_bias = self.get_attr('quantizer')(input_transform_bias)
        data = np.dot(input_transform_bias, output_transform_kernel) # [n_aggregators, n_out_features]
        bias = data.transpose((1, 0))

        return kernel, bias

    def _add_variable(self, name, var_name, data, frac_width=10, quantize=False):
        # Wrapper for add_weights_variable with precision determination from data

        # automatically make the variable unsigned if data are all positive
        signed = (np.amin(data) < 0.)
        
        int_width = find_minimum_width(data, signed=signed)

        if quantize:
            precision = IntegerPrecisionType(width=int_width, signed=signed)
        else:
            width = int_width + frac_width
            precision = FixedPrecisionType(width=width, integer=int_width, signed=signed, rounding_mode='AP_RND', saturation_mode='AP_SAT')
            
        self.add_weights_variable(name=name, var_name=var_name, data=data, precision=precision)
        
    def function_cpp(self):
        params = self._default_function_params()

        data = self.get_input_variable(self.inputs[0])
        integer_input = self.get_input_variable(self.inputs[1])
        params['input_t'] = data.type.name
        params['input'] = data.name

        params['integer_input_t'] = integer_input.type.name
        params['nvtx'] = integer_input.name

        if self.ref_impl:
            params['impl'] = '_ref'
        else:
            params['impl'] = ''

        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()

        params['n_vertices'] = self.attributes['n_vertices']
        params['n_vertices_width'] = int(np.log2(params['n_vertices']))
        params['distance_width'] = 12
        params['distance_nint'] = min(4, params['distance_width'] - 6) # this is tuned
        params['log2_reuse'] = int(np.log2(params['reuse']))

        ## Define default precisions for various internal arrays (can be overridden from the config file)
        # We always give 10 digits for the subintegral part
        fwidth = 10
        # Integral precision for aggr_t depends on how large the temporary sum for weighed feature mean will be
        aggr_intw = max(params['log2_reuse'], params['n_vertices_width'] - params['log2_reuse']) + 3 # safety factor 2**3
        aggr_w = aggr_intw + fwidth
        # edge_weight_aggr_t does not need the safety factor
        ew_aggr_intw = aggr_intw - 3
        ew_aggr_w = ew_aggr_intw + fwidth
        # Integral precision for norm is fixed to 4
        norm_intw = 4
        norm_w = norm_intw + fwidth

        vspecs = [
            ('edge_weight', FixedPrecisionType(10, 0, signed=False)),
            ('edge_weight_aggr', FixedPrecisionType(ew_aggr_w, ew_aggr_intw, signed=False)),
            ('aggr', FixedPrecisionType(aggr_w, aggr_intw)),
            ('norm', FixedPrecisionType(norm_w, norm_intw, signed=False))
        ]
        for vname, default_precision in vspecs:
            params['{}_t'.format(vname)], type_name = self.model.config.get_precision(self, var=vname)
            if type_name.endswith('default_t'):
                params['{}_t'.format(vname)] = str(default_precision)

        params['output_t'] = self.get_output_variable().type.name

        if self.attributes['collapse'] in ['mean', 'max']:
            params['collapse_type'] = 'collapse_{}'.format(self.attributes['collapse'])
        else:
            params['collapse_type'] = 'no_collapse'

        params['mean_by_nvert'] = str(self.attributes['mean_by_nvert']).lower()

        self._get_transforms_config(params)

        return self._config_template.format(**params)

    def _get_transforms_config(self, params):
        params['n_in_features'] = self.attributes['n_in_features']
        params['n_propagate'] = self.attributes['n_propagate']
        params['n_aggregators'] = self.get_weights('aggregator_distance_biases').shape[0]
        params['n_out_features'] = self.get_weights('output_transform_biases').shape[0]

        for wname, weights in self.weights.items():
            params[wname] = weights.name
            params['{}_t'.format(wname)] = weights.type.name
            params['{}_size'.format(wname)] = weights.data_length


class GarNetStack(GarNet):
    def _initialize_transforms(self):
        self._sublayer_weights = []

        quantize = (self.get_attr('quantizer') is not None)

        for il in range(self.attributes['n_sublayers']):
            sublayer_weights = {}

            n_aggregators = self.attributes['n_aggregators'][il]
            n_out_features = self.attributes['n_out_features'][il]
            n_propagate = self.attributes['n_propagate'][il]

            kernel, bias = self._make_input_transform_weights(n_propagate, n_aggregators, n_out_features, quantize=quantize, sublayer=il)

            name = 'input_transform_{}_weights'.format(il)
            self._add_variable(name, 'input_transform_{}_w{{index}}'.format(il), kernel, frac_width=10, quantize=quantize)
            sublayer_weights['input_transform_weights'] = self.weights[name]

            name = 'input_transform_{}_biases'.format(il)
            self._add_variable(name, 'input_transform_{}_b{{index}}'.format(il), bias, frac_width=10, quantize=quantize)
            sublayer_weights['input_transform_biases'] = self.weights[name]
        
            weights_source = [
                ('aggregator_distance', 'S{}'.format(il), 'kernel'),
                ('aggregator_distance', 'S{}'.format(il), 'bias'),
                ('output_transform', 'Fout{}'.format(il), 'bias')
            ]
    
            for op_name, lname, wtype in weights_source:
                data = self.model.get_weights_data(self.name, '{name}/{lname}_{wtype}:0'.format(name=self.name, lname=lname, wtype=wtype))
                if wtype == 'kernel':
                    data = data.transpose((1, 0))
                    vtype = 'weights'
                else:
                    vtype = 'biases'

                name = '{}_{}_{}'.format(op_name, il, vtype)
                var_name = '{}_{}_{}{{index}}'.format(op_name, il, vtype[0])

                self._add_variable(name, var_name, data, frac_width=10, quantize=False)
                sublayer_weights['{}_{}'.format(op_name, vtype)] = self.weights[name]

            self._sublayer_weights.append(sublayer_weights)

        self._output_features = self.attributes['n_out_features'][-1]

    def _get_transforms_config(self, params):
        base_template, sublayer_template = self._config_template
        self._config_template = base_template

        params['n_sublayers'] = self.attributes['n_sublayers']
        params['n_in_features'] = self.attributes['n_in_features'][0]
        params['n_out_features'] = self.attributes['n_out_features'][-1]

        sublayer_configs = []
        for il in range(self.attributes['n_sublayers'] - 1, -1, -1):
            sub_params = {'index': self.index, 'il': il}

            for p in ['n_in_features', 'n_propagate', 'n_aggregators', 'n_out_features']:
                sub_params[p] = self.attributes[p][il]

            for wname, weights in self._sublayer_weights[il].items():
                sub_params[wname] = weights.name
                sub_params['{}_t'.format(wname)] = weights.type.name
                sub_params['{}_size'.format(wname)] = weights.data_length

            if il != self.attributes['n_sublayers'] - 1:
                sub_params['next'] = il + 1
            else:
                sub_params['next'] = 0

            sublayer_configs.append(sublayer_template.format(**sub_params))

        params['sublayer_configs'] = '\n'.join(sublayer_configs)

layer_map = {
    'InputLayer'         : Input,
    'Activation'         : Activation,
    'QActivation'        : Activation,
    'LeakyReLU'          : ParametrizedActivation,
    'ThresholdedReLU'    : ParametrizedActivation,
    'ELU'                : ParametrizedActivation,
    'PReLU'              : PReLU,
    'Softmax'            : Softmax,
    'Reshape'            : Reshape,
    'Dense'              : Dense,
    'BinaryDense'        : Dense,
    'TernaryDense'       : Dense,
    'QDense'             : Dense,
    'Conv1D'             : Conv1D,
    'QConv1D'            : Conv1D,
    'Conv2D'             : Conv2D,
    'BinaryConv2D'       : Conv2D,
    'QConv2D'            : Conv2D,
    'BatchNormalization' : BatchNormalization,
    'MaxPooling1D'       : Pooling1D,
    'AveragePooling1D'   : Pooling1D,
    'MaxPooling2D'       : Pooling2D,
    'AveragePooling2D'   : Pooling2D,
    'Merge'              : Merge,
    'Concatenate'        : Concatenate,
    'Resize'             : Resize,
    'Transpose'          : Transpose,
    'GarNet'             : GarNet,
    'GarNetStack'        : GarNetStack,
    # TensorFlow-specific layers:
    'BiasAdd'            : BiasAdd,
}

def register_layer(name, clazz):
    global layer_map
    layer_map[name] = clazz
