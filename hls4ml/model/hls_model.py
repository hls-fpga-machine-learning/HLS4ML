from __future__ import print_function
import six
import re
import numpy as np
from collections import OrderedDict

from ..templates import get_backend
from ..writer import get_writer

class HLSConfig(object):
    def __init__(self, config):
        self.config = config

        self.backend = get_backend(self.config.get('Backend', 'Vivado'))
        self.writer = get_writer(self.config.get('Backend', 'Vivado'))

        self.model_precision = {}
        self.layer_type_precision = {}
        self.layer_name_precision = {}

        self.model_rf = None
        self.layer_type_rf = {}
        self.layer_name_rf = {}

        self.model_strategy = 'Latency'
        self.layer_type_strategy = {}
        self.layer_name_strategy = {}

        self.model_compression = False
        self.layer_type_compression = {}
        self.layer_name_compression = {}

        self._parse_hls_config()
        self._validate_hls_config()

    def get_config_value(self, key, default=None):
        return self.config.get(key, default)

    def get_project_name(self):
        return self.get_config_value('ProjectName')

    def get_output_dir(self):
        return self.get_config_value('OutputDir')

    def get_layer_config_value(self, layer, key, default=None):
        hls_config = self.config['HLSConfig']

        name_config = hls_config.get('LayerName', {}).get(layer.name.lower(), None)
        if name_config is not None:
            return name_config.get(key, default)
        
        type_config = hls_config.get('LayerType', {}).get(layer.__class__.__name__, None)
        if type_config is not None:
            return type_config.get(key, default)
        
        return default

    def get_precision(self, layer, var='default'):
        precision = self.layer_name_precision.get(layer.name.lower() + '_' + var)
        type_name = layer.name.lower() + '_' + var + '_t'
        if precision is None:
            precision = self.layer_name_precision.get(layer.name.lower() + '_default')
            type_name = layer.name.lower() + '_default_t'

        if precision is None:
            precision = self.layer_type_precision.get(layer.__class__.__name__.lower() + '_' + var)
            type_name = layer.__class__.__name__ + '_' + var + '_t'
        if precision is None:
            precision = self.layer_type_precision.get(layer.__class__.__name__.lower() + '_default')
            type_name = layer.__class__.__name__ + '_default_t'

        if precision is None:
            precision = self.model_precision.get(var)
            type_name = var + '_default_t'
        if precision is None:
            precision = self.model_precision.get('default')
            type_name = 'model_default_t'

        if precision is None:
            raise Exception('No precision for {}->{} found and no default specified.'.format(layer.name, var))

        return (precision, type_name)

    def get_reuse_factor(self, layer):
        rf = self.layer_name_rf.get(layer.name.lower())
        if rf is None:
            rf = self.layer_type_rf.get(layer.__class__.__name__.lower())
        if rf is None:
            rf = self.model_rf

        if rf is None:
            raise Exception('No reuse factor for {} found and no default specified.'.format(layer.name))

        return rf

    def get_strategy(self, layer):
        strategy = self.layer_name_strategy.get(layer.name.lower())
        if strategy is None:
            strategy = self.layer_type_strategy.get(layer.__class__.__name__.lower())
        if strategy is None:
            strategy = self.model_strategy

        return strategy

    def is_resource_strategy(self, layer):
        return self.get_strategy(layer).lower() == 'resource'

    def get_compression(self, layer):
        compression = self.layer_name_compression.get(layer.name.lower())
        if compression is None:
            compression = self.layer_type_compression.get(layer.__class__.__name__.lower())
        if compression is None:
            compression = self.model_compression

        return compression

    def _parse_hls_config(self):
        hls_config = self.config['HLSConfig']
        model_cfg = hls_config.get('Model')
        if model_cfg is not None:
            precision_cfg = model_cfg.get('Precision')
            if precision_cfg is not None:
                if isinstance(precision_cfg, dict):
                    for var, precision in precision_cfg.items():
                        self.model_precision[var] = precision
                else:
                    self.model_precision['default'] = precision_cfg # Default precision for everything

            self.model_rf = model_cfg.get('ReuseFactor')
            self.model_strategy = model_cfg.get('Strategy', 'Latency')
            self.model_compression = bool(model_cfg.get('Compression', 0))

        layer_type_cfg = hls_config.get('LayerType')
        if layer_type_cfg is not None:
            for layer_type, layer_cfg in layer_type_cfg.items():
                precision_cfg = layer_cfg.get('Precision')
                if isinstance(precision_cfg, dict):
                    for var, precision in precision_cfg.items():
                        self.layer_type_precision[layer_type.lower() + '_' + var] = precision
                else:
                    self.layer_type_precision[layer_type.lower() + '_default'] = precision_cfg

                rf = layer_cfg.get('ReuseFactor')
                if rf is not None:
                    self.layer_type_rf[layer_type.lower()] = rf

                strategy = layer_cfg.get('Strategy')
                if strategy is not None:
                    self.layer_type_strategy[layer_type.lower()] = strategy

                compression = layer_cfg.get('Compression')
                if compression is not None:
                    self.layer_type_compression[layer_type.lower()] = bool(compression)

        layer_name_cfg = hls_config.get('LayerName')
        if layer_name_cfg is not None:
            for layer_name, layer_cfg in layer_name_cfg.items():
                precision_cfg = layer_cfg.get('Precision')
                if isinstance(precision_cfg, dict):
                    for var, precision in precision_cfg.items():
                        self.layer_name_precision[layer_name.lower() + '_' + var] = precision
                else:
                    self.layer_name_precision[layer_name.lower() + '_default'] = precision_cfg

                rf = layer_cfg.get('ReuseFactor')
                if rf is not None:
                    self.layer_name_rf[layer_name.lower()] = rf

                strategy = layer_cfg.get('Strategy')
                if strategy is not None:
                    self.layer_name_strategy[layer_name.lower()] = strategy

                compression = layer_cfg.get('Compression')
                if compression is not None:
                    self.layer_name_compression[layer_name.lower()] = bool(compression)

    def _validate_hls_config(self):
        use_resource = False
        if self.model_strategy.lower() == 'latency' and self.model_compression:
            print('WARNING: Compression enabled while model strategy set to "Latency".')
            use_resource = True
        for layer_type, strategy in self.layer_type_strategy.items():
            if strategy.lower() == 'resource' and self.model_strategy.lower() == 'latency':
                print('WARNING: Strategy for layer type {} set to "Resource", while model strategy set to "Latency".'.format(layer_type))
                use_resource = True

        for layer_name, strategy in self.layer_name_strategy.items():
            if strategy.lower() == 'resource' and self.model_strategy.lower() == 'latency':
                print('WARNING: Strategy for layer {} set to "Resource", while model strategy set to "Latency".'.format(layer_name))
                use_resource = True

        for layer_type, compression in self.layer_type_compression.items():
            if compression and self.model_strategy.lower() == 'latency':
                print('WARNING: Compression enabled for layer type {}, while model strategy set to "Latency".'.format(layer_type))
                use_resource = True

        for layer_name, compression in self.layer_name_compression.items():
            if compression and self.model_strategy.lower() == 'latency':
                print('WARNING: Compression enabled for layer {}, while model strategy set to "Latency".'.format(layer_name))
                use_resource = True

        if use_resource:
            print('WARNING: Changing model strategy to "Resource"')
            self.model_strategy = 'Resource'

class HLSModel(object):
    def __init__(self, config, data_reader, layer_list, inputs=None, outputs=None):
        self.config = HLSConfig(config)
        self.reader = data_reader

        # If not provided, assumes layer_list[0] is input, and layer_list[-1] is output
        self.inputs = inputs if inputs is not None else [layer_list[0]['name']]
        self.outputs = outputs if outputs is not None else [layer_list[-1]['name']]

        self.index = 0
        self.graph = OrderedDict()
        self.output_vars = {}

        self._make_graph(layer_list)

    def _make_graph(self, layer_list):
        for layer in layer_list:
            kind = layer['class_name']
            name = layer['name']
            inputs = layer.get('inputs', [])
            outputs = layer.get('outputs', [])
            if len(inputs) == 0:
                inputs = [next(reversed(self.graph), 'input')]
            if len(outputs) == 0:
                outputs = [name]

            self.graph[name] = self.make_node(kind, name, layer, inputs, outputs)

    def make_node(self, kind, name, attributes, inputs, outputs=None):
        node = layer_map[kind](self, name, attributes, inputs, outputs)
        for o in node.outputs:
            out_var = node.get_output_variable(output_name=o)
            if o in self.outputs:
                out_var.type.name = 'result_t'
            self.output_vars[o] = out_var

        return node

    def insert_node(self, node):
        if len(node.inputs) > 1:
            raise Exception('Cannot insert a node with more than one input (for now).')

        prev_node = self.graph.get(node.inputs[0])
        next_node = next((x for x in self.graph.values() if x.inputs[0] == prev_node.outputs[0]), None)
        if next_node is not None:
            next_node.inputs[0] = node.outputs[0]

        new_graph = OrderedDict()
        for k, v in self.graph.items():
            new_graph[k] = v
            if k == prev_node.name:
                new_graph[node.name] = node

        self.graph = new_graph

    def remove_node(self, node, rewire=True):
        if rewire:
            if len(node.inputs) > 1 or len(node.outputs) > 1:
                raise Exception('Cannot rewire a node with multiple inputs/outputs')
            prev_node = self.graph.get(node.inputs[0])
            next_node = next((x for x in self.graph.values() if x.inputs[0] == node.outputs[0]), None)
            if prev_node is not None:
                if next_node is not None:
                    next_node.inputs[0] = prev_node.outputs[0]
                else:
                    if node.outputs[0] in self.outputs:
                        self.outputs = [prev_node.outputs[0] if x == node.outputs[0] else x for x in self.outputs]
                    else:
                        raise Exception('Cannot rewire a node without child')
            else:
                raise Exception('Cannot rewire a node without a parent')

        del self.output_vars[node.outputs[0]]
        del self.graph[node.name]

    def replace_node(self, old_node, new_node):
        prev_node = self.graph.get(old_node.inputs[0])
        next_node = next((x for x in self.graph.values() if x.inputs[0] == old_node.outputs[0]), None)
        if next_node is not None:
            next_node.inputs[0] = new_node.outputs[0]
        if prev_node is not None:
            if new_node.inputs is None or len(new_node.inputs) == 0: # Check if already rewired
                new_node.inputs = [prev_node.outputs[0]]

        self.graph = OrderedDict((new_node.name, new_node) if k == old_node.name else (k, v) for k, v in self.graph.items())

    def get_weights_data(self, layer_name, var_name):
        return self.reader.get_weights_data(layer_name, var_name)

    def quantize_data(self, data, quantize):
        zeros = np.zeros_like(data)
        ones = np.ones_like(data)
        quant_data = data
        if quantize == 1:
            quant_data = np.where(data > 0, ones, zeros).astype('int')
        if quantize == 2:
            quant_data = np.where(data > 0, ones, -ones)
        elif quantize == 3:
            quant_data = np.where(data > 0.5, ones, np.where(data <= -0.5, -ones, zeros))
        return quant_data

    def next_layer(self):
        self.index += 1
        return self.index

    def get_layers(self):
        return self.graph.values()

    def get_input_variables(self):
        variables = []
        for inp in self.inputs:
            variables.append(self.graph[inp].get_output_variable())
        return variables

    def register_output_variable(self, out_name, variable):
        if out_name in self.outputs:
            variable.type.name = 'result_t'
        self.output_vars[out_name] = variable

    def get_output_variables(self):
        variables = []
        for out in self.outputs:
            variables.append(self.output_vars[out])
        return variables

    def get_layer_output_variable(self, output_name):
        return self.output_vars[output_name]

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
    def __init__(self, var_name, type_name, precision, data, **kwargs):
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
    def __init__(self, var_name, type_name, precision, data, reuse_factor, **kwargs):
        super(CompressedWeightVariable, self).__init__(var_name, type_name, precision, data, **kwargs)
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
        self.type = CompressedType(type_name, precision, 'ap_uint<{}>'.format(index_precision), **kwargs)

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

        self._function_template = self.model.config.backend.get_function_template(self.__class__.__name__)
        self._config_template = self.model.config.backend.get_config_template(self.__class__.__name__)
        self.include_list = self.model.config.backend.get_include_list(self.__class__.__name__)
        self.weights = OrderedDict()
        self.variables = OrderedDict()
        self.precision = OrderedDict()
        accum_t = HLSType(*reversed(self.model.config.get_precision(self, 'accum')))
        self.precision[accum_t.name] = accum_t
        self.set_attr('accum_t', accum_t.precision)
        self.reuse_factor = self.model.config.get_reuse_factor(self)

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

    def get_output_variable(self, output_name=None):
        if output_name is not None:
            return self.variables[output_name]
        else:
            return next(iter(self.variables.values()))

    def get_weights(self, var_name=None):
        if var_name:
            return self.weights[var_name]

        return self.weights.values()

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

    def add_weights(self, quantize=0, compression=False):
        data = self.model.get_weights_data(self.name, 'kernel')

        self.add_weights_variable(name='weight', var_name='w{index}', data=data, quantize=quantize, compression=compression)

    def add_bias(self, quantize=0):
        data = self.model.get_weights_data(self.name, 'bias')
        precision = None
        type_name = None
        if data is None:
            data = np.zeros(self.get_output_variable().shape[-1])
            precision = 'ap_uint<1>'
            type_name = 'bias{index}_t'
            quantize = 0 # Don't quantize non-existant bias

        self.add_weights_variable(name='bias', var_name='b{index}', type_name=type_name, precision=precision, data=data, quantize=quantize)

    def add_weights_variable(self, name, var_name=None, type_name=None, precision=None, data=None, quantize=0, compression=False):
        if var_name is None:
            var_name = name + '{index}'

        if precision is None:
            precision, _ = self.model.config.get_precision(self, var=name)

        if type_name is None:
            _, type_name = self.model.config.get_precision(self, var=name)

        if data is None:
            data = self.model.get_weights_data(self.name, name)
        elif isinstance(data, six.string_types):
            data = self.model.get_weights_data(self.name, data)

        if quantize > 0:
            data = self.model.quantize_data(data, quantize)
            if quantize == 1:
                precision = 'ap_uint<1>'
                type_name = name + '{index}_t'
            elif quantize == 2 or quantize == 3:
                precision = 'ap_int<2>'
                type_name = name + '{index}_t'

        if compression:
            var = CompressedWeightVariable(var_name, type_name=type_name, precision=precision, data=data, reuse_factor=self.reuse_factor, index=self.index)
        else:
            var = WeightVariable(var_name, type_name=type_name, precision=precision, data=data, index=self.index)

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
        quantize = self.get_attr('quantize', default=0)
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
        self.add_weights(quantize=quantize, compression=compression)
        index_t = 'ap_uint<1>'
        if self.model.config.is_resource_strategy(self):
            if self.model.config.get_compression(self):
                index_t = self.get_weights('weight').type.index_precision
            else:
                if self.model.config.backend.name == 'Vivado':
                    self.weights['weight'].data = np.transpose(self.weights['weight'].data)
        self.set_attr('index_t', index_t)
        self.add_bias(quantize=quantize)

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

class Conv1D(Layer):
    def initialize(self):
        if self.get_attr('data_format') == 'channels_last':
            shape = [self.attributes['n_out'], self.attributes['n_filt']]
            dims = ['N_OUTPUTS_{}'.format(self.index), 'N_FILT_{}'.format(self.index)]
        else:
            shape = [self.attributes['n_filt'], self.attributes['n_out']]
            dims = ['N_FILT_{}'.format(self.index), 'N_OUTPUTS_{}'.format(self.index)]
        
        self.add_output_variable(shape, dims)
        self.add_weights()
        self.add_bias()
        if self.model.config.is_resource_strategy(self):
            self.set_attr('strategy', 'large')
            if self.model.config.backend.name == 'Vivado':
                self.model.config.backend.set_closest_reuse_factor(self)
                self.weights['weight'].data = np.transpose(self.weights['weight'].data, axes=[2, 1, 0]) #(W,C,F) => (F,C,W)
        else:
            self.set_attr('strategy', 'latency')

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

class Conv2D(Layer):
    def initialize(self):
        if self.get_attr('data_format') == 'channels_last':
            shape = [self.attributes['out_height'], self.attributes['out_width'], self.attributes['n_filt']]
            dims = ['OUT_HEIGHT_{}'.format(self.index), 'OUT_WIDTH_{}'.format(self.index), 'N_FILT_{}'.format(self.index)]
        else:
            shape = [self.attributes['n_filt'], self.attributes['out_height'], self.attributes['out_width']]
            dims = ['N_FILT_{}'.format(self.index), 'OUT_HEIGHT_{}'.format(self.index), 'OUT_WIDTH_{}'.format(self.index)]
        self.add_output_variable(shape, dims)
        self.add_weights()
        self.add_bias()
        if self.model.config.is_resource_strategy(self):
            self.set_attr('strategy', 'large')
            if self.model.config.backend.name == 'Vivado':
                self.model.config.backend.set_closest_reuse_factor(self)
                self.weights['weight'].data = np.transpose(self.weights['weight'].data, axes=[3, 2, 0, 1]) #(H,W,C,F) => (F,C,H,W)
        else:
            self.set_attr('strategy', 'latency')

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

class LSTM(Layer):
    def initialize(self):
        shape = [self.attributes['n_sequence_out'], int(self.attributes['recurr_n_out']/4)]
        dims = ['N_SEQUENCE_OUT_{}'.format(self.index), 'N_LAYER_{}'.format(self.index)]        
        self.add_output_variable(shape, dims)
        self.add_weights()
        self.add_bias()
        recurrent_weight = self.model.get_weights_data(self.name, 'recurrent_kernel')
        self.add_weights_variable(name='recurrent_weight', var_name='wr{index}', data=recurrent_weight)
        recurrent_bias = np.zeros(self.get_output_variable().shape[-1]*4)
        self.add_weights_variable(name='recurrent_bias', var_name='br{index}', data=recurrent_bias)

		
    def function_cpp(self):
        params = self._default_function_params()
        params['w'] = self.get_weights('weight').name
        params['b'] = self.get_weights('bias').name
        params['wr'] = self.get_weights('recurrent_weight').name
        params['br'] = self.get_weights('recurrent_bias').name
        params['activation'] = self.get_attr('activation')
        params['recurrent_activation'] = self.get_attr('recurrent_activation')

        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()
        params['n_in'] = self.get_input_variable().dim_names[1]
        params['n_sequence'] = self.get_input_variable().dim_names[0]
        params['n_sequence_out'] = self.attributes['n_sequence_out']
        params['n_state'] = self.get_output_variable().dim_names[1]
        params['n_out'] = self.get_output_variable().dim_names[1]
        params['nzeros'] = self.get_weights('weight').nzeros

        params['config_mult_t1'] = 'config{}_mult1'.format(self.index)
        params['config_mult_t2'] = 'config{}_mult2'.format(self.index)
        params['lstm_act_t'] = '{}_config{}_recr'.format(self.get_attr('recurrent_activation'), self.index)
        params['act_t'] = '{}_config{}'.format(self.get_attr('activation'), self.index)
        lstm_config = self._config_template[0].format(**params)
        

        lstm_params = self._default_config_params()
        lstm_params['n_in'] = self.get_output_variable().dim_names[1] + ' * 3'
        lstm_act_config = self._config_template[2].format(**lstm_params)

        act_params = self._default_config_params()
        act_params['n_in'] = self.get_output_variable().dim_names[1]
        act_params['table_size'] = 1024
        act_params['table_t'] = 'ap_fixed<18,8>'
        act_config = self._config_template[3].format(**act_params)

        mult_params1 = self._default_config_params()
        mult_params2 = self._default_config_params()
        mult_params1['n_in'] = self.get_input_variable().dim_names[1]
        mult_params1['n_out'] = self.get_output_variable().dim_names[1] + ' * 4'
        mult_params2['n_in'] = self.get_output_variable().dim_names[1]
        mult_params2['n_out'] = self.get_output_variable().dim_names[1] + ' * 4'

        mult_config1 = self._config_template[1].format(**mult_params1)
        mult_config2 = self._config_template[4].format(**mult_params2)


        return mult_config1 + '\n' + mult_config2 + '\n' + lstm_act_config + '\n' + act_config + '\n' + lstm_config

class GRU(Layer):
    def initialize(self):
        shape = [self.attributes['n_sequence_out'],int(self.attributes['recurr_n_out']/3)]
        dims = ['N_SEQUENCE_OUT_{}'.format(self.index), 'N_LAYER_{}'.format(self.index)]
        self.add_output_variable(shape, dims)
        self.add_weights()
        self.add_bias()
        recurrent_weight = self.model.get_weights_data(self.name, 'recurrent_kernel')
        self.add_weights_variable(name='recurrent_weight', var_name='wr{index}', data=recurrent_weight)
        recurrent_bias = np.zeros(self.get_output_variable().shape[-1]*3)
        self.add_weights_variable(name='recurrent_bias', var_name='br{index}', data=recurrent_bias)
		
    def function_cpp(self):
        params = self._default_function_params()
        params['w'] = self.get_weights('weight').name
        params['b'] = self.get_weights('bias').name
        params['wr'] = self.get_weights('recurrent_weight').name
        params['br'] = self.get_weights('recurrent_bias').name
        params['activation'] = self.get_attr('activation')
        params['recurrent_activation'] = self.get_attr('recurrent_activation')

        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()
        params['n_in'] = self.get_input_variable().dim_names[1]
        params['n_sequence'] = self.get_input_variable().dim_names[0]
        params['n_sequence_out'] = self.attributes['n_sequence_out']
        params['n_state'] = self.get_output_variable().dim_names[1]
        params['n_out'] = self.get_output_variable().dim_names[1]
        params['nzeros'] = self.get_weights('weight').nzeros

        params['config_mult_t1'] = 'config{}_mult1'.format(self.index)
        params['config_mult_t2'] = 'config{}_mult2'.format(self.index)
        params['gru_act_t'] = '{}_config{}_recr'.format(self.get_attr('recurrent_activation'), self.index)
        params['act_t'] = '{}_config{}'.format(self.get_attr('activation'), self.index)
        gru_config = self._config_template[0].format(**params)
        
        gru_params = self._default_config_params()
        gru_params['n_in'] = self.get_output_variable().dim_names[1] + ' * 2'
        gru_act_config = self._config_template[2].format(**gru_params)

        act_params = self._default_config_params()
        act_params['n_in'] = self.get_output_variable().dim_names[1]
        act_params['table_size'] = 1024
        act_params['table_t'] = 'ap_fixed<18,8>'
        act_config = self._config_template[3].format(**act_params)

        mult_params1 = self._default_config_params()
        mult_params2 = self._default_config_params()
        mult_params1['n_in'] = self.get_input_variable().dim_names[1] # n_in -> n_in * 3
        mult_params1['n_out'] = self.get_output_variable().dim_names[1] + ' * 3'
        mult_params2['n_in'] = self.get_output_variable().dim_names[1] # n_state -> n_state *3
        mult_params2['n_out'] = self.get_output_variable().dim_names[1] + ' * 3'

        mult_config1 = self._config_template[1].format(**mult_params1)
        mult_config2 = self._config_template[4].format(**mult_params2)

        return mult_config1 + '\n' + mult_config2 + '\n' + gru_act_config + '\n' + act_config + '\n' + gru_config

class Pooling1D(Layer):
    def initialize(self):
        shape = [self.attributes['n_out'], self.attributes['n_filt']]
        dims = ['N_OUTPUTS_{}'.format(self.index), 'N_FILT_{}'.format(self.index)]
        self.add_output_variable(shape, dims)
        self.set_attr('pool_op', self.get_attr('class_name').split('Pooling')[0])

    def function_cpp(self):
        params = self._default_function_params()

        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()
        params['n_in'] = self.get_input_variable().size_cpp()
        params['n_out'] = self.get_output_variable().size_cpp()

        return self._config_template.format(**params)

class Pooling2D(Layer):
    def initialize(self):
        shape = [self.attributes['out_height'], self.attributes['out_width'], self.attributes['n_filt']]
        dims = ['OUT_HEIGHT_{}'.format(self.index), 'OUT_WIDTH_{}'.format(self.index), 'N_FILT_{}'.format(self.index)]
        self.add_output_variable(shape, dims)
        self.set_attr('pool_op', self.get_attr('class_name').split('Pooling')[0])

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

class Activation(Layer):
    def initialize(self):
        inp = self.get_input_variable()
        shape = inp.shape
        dims = inp.dim_names
        self.add_output_variable(shape, dims)
        if self.model.config.backend.name == 'Vivado':
            self.set_attr('table_t', self.model.config.get_layer_config_value(self, 'table_t', 'ap_fixed<18,8>'))
            self.set_attr('table_size', self.model.config.get_layer_config_value(self, 'table_size', 1024))

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
        shape = [sum(x) for x in zip(inp1.shape, inp2.shape)]
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

layer_map = {
    'InputLayer'         : Input,
    'Activation'         : Activation,
    'QActivation'        : Activation,
    'LeakyReLU'          : ParametrizedActivation,
    'ThresholdedReLU'    : ParametrizedActivation,
    'ELU'                : ParametrizedActivation,
    'PReLU'              : PReLU,
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
    'LSTM'               : LSTM,
    'GRU'                : GRU,
    'BatchNormalization' : BatchNormalization,
    'MaxPooling1D'       : Pooling1D,
    'AveragePooling1D'   : Pooling1D,
    'MaxPooling2D'       : Pooling2D,
    'AveragePooling2D'   : Pooling2D,
    'Merge'              : Merge,
    'Concatenate'        : Concatenate,
    'Resize'             : Resize,
    'Transpose'          : Transpose,
    # TensorFlow-specific layers:
    'BiasAdd'            : BiasAdd,
}

def register_layer(name, clazz):
    global layer_map
    layer_map[name] = clazz
