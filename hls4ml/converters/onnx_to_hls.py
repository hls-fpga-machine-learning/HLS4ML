from __future__ import print_function
import numpy as np
import math
from onnx import ModelProto, GraphProto, NodeProto, TensorProto
from onnx import optimizer, helper, numpy_helper, shape_inference

from hls4ml.model import HLSModel

MAXMULT = 4096


class ONNXDataReader:
    def __init__(self, model):
        self.model = model
        self.input_map = {}
        self.index_map = {
            # Dense
            'kernel': 1,
            'bias': 2,
            # BatchNormalization
            'gamma': 1,
            'beta': 2,
            'moving_mean': 3,
            'moving_variance': 4,
        }

    def get_weights_data(self, layer_name, var_name):
        inputs = self.input_map[layer_name]
        inp_idx = self.index_map[var_name]
        if inp_idx >= len(inputs['inputs']):
            # Input not found, likely a bias tensor is not available
            return None

        tensor = next((x for x in self.model.graph.initializer if x.name == inputs['inputs'][inp_idx]), None)
        if tensor is not None:
            data = numpy_helper.to_array(tensor)
            if inputs['transpose']:
                if inputs['perm'] is not None and len(data.shape) == len(inputs['perm']):
                    data = data.transpose(inputs['perm'])
                else:
                    data = data.transpose()

        return data

    def add_input(self, layer_name, inputs, transpose=True, perm=None):
        self.input_map[layer_name] = {'inputs': inputs, 'transpose': transpose, 'perm': perm}


def sanitize_layer_name(layer):
    new_name = layer['name']
    if new_name[0].isdigit():
        new_name = layer['class_name'].lower() + new_name

    layer['name'] = new_name


def get_onnx_attribute(operation, name, default=None):
    attr = next((x for x in operation.attribute if x.name == name), None)
    if attr is None:
        value = default
    else:
        value = helper.get_attribute_value(attr)
        if isinstance(value, bytes):
            value = value.decode()
    return value


def get_input_shape(model, operation, input_idx=0):
    value_info_idx = next((i for i, x in enumerate(model.graph.value_info) if x.name == operation.input[input_idx]), 0)
    return [d.dim_value for d in model.graph.value_info[value_info_idx].type.tensor_type.shape.dim]


def compute_pads_1d(operation, layer):
    auto_pad = get_onnx_attribute(operation, 'auto_pad', 'NOTSET')
    if auto_pad != 'NOTSET':
        if (layer['y_in'] % layer['stride'] == 0):
            pad_along_width = max(layer['y_filt'] - layer['stride'], 0)
        else:
            pad_along_width = max(layer['y_filt'] - (layer['y_in'] % layer['stride']), 0)

        pads = [pad_along_width // 2, pad_along_width - (pad_along_width // 2)]

        if auto_pad == 'SAME_UPPER':
            pads = sorted(pads)
        elif auto_pad == 'SAME_LOWER':
            pads = sorted(pads, reverse=True)
        else:  # 'VALID' padding
            pads = [0, 0]
    else:
        pads = get_onnx_attribute(operation, 'pads', [0, 0])

    return pads


def compute_pads_2d(operation, layer):
    auto_pad = get_onnx_attribute(operation, 'auto_pad', 'NOTSET')
    if auto_pad != 'NOTSET':
        # Height
        if (layer['in_height'] % layer['stride_height'] == 0):
            pad_along_height = max(layer['filt_height'] - layer['stride_height'], 0)
        else:
            pad_along_height = max(layer['filt_height'] - (layer['in_height'] % layer['stride_height']), 0)
        pad_height = [pad_along_height // 2, pad_along_height - pad_along_height // 2]

        # Width
        if (layer['in_width'] % layer['stride_width'] == 0):
            pad_along_width = max(layer['filt_width'] - layer['stride_width'], 0)
        else:
            pad_along_width = max(layer['filt_width'] - (layer['in_width'] % layer['stride_width']), 0)
        pad_width = [pad_along_width // 2, pad_along_width - pad_along_width // 2]

        if auto_pad == 'SAME_UPPER':
            pads = [min(pad_height), min(pad_width), max(pad_height), max(pad_width)]
        elif auto_pad == 'SAME_LOWER':
            pads = [max(pad_height), max(pad_width), min(pad_height), min(pad_width)]
        else:  # 'VALID' padding
            pads = [0, 0, 0, 0]
    else:
        pads = get_onnx_attribute(operation, 'pads', [0, 0, 0, 0])

    return pads


def onnx_to_hls(yamlConfig):

    ######################
    # Do translation
    ######################

    # This is a list of dictionaries to hold all the layer info we need to generate HLS
    layer_list = []

    # Extract model architecture
    model = ModelProto()
    with open(yamlConfig['OnnxModel'], 'rb') as fid:
        model.ParseFromString(fid.read())

    # Define supported layers
    core_operations = ['Gemm', 'BatchNormalization', 'Conv']
    transform_operations = ['Squeeze', 'Unsqueeze', 'Transpose', 'Flatten', 'Identity', 'Reshape']
    pool_operations = ['AveragePool', 'MaxPool']
    merge_operations = ['Add', 'Sub', 'Mul', 'Average', 'Max', 'Min', 'Concat', 'Sum']
    activation_operations = ['Relu', 'Tanh', 'Sigmoid', 'LeakyRelu', 'ThresholdedRelu', 'HardSigmoid', 'Elu', 'Selu', 'PRelu', 'Softmax', 'Softsign', 'Softplus']
    supported_operations = core_operations + transform_operations + pool_operations + merge_operations + activation_operations

    operation_map = {'Gemm': 'Dense', 'Relu': 'Activation', 'Tanh': 'Activation', 'Sigmoid': 'Activation',
                     'LeakyRelu': 'LeakyReLU', 'ThresholdedRelu': 'ThresholdedReLU', 'HardSigmoid': 'Activation',
                     'Elu': 'ELU', 'Selu': 'Activation', 'PRelu': 'PReLU', 'Softmax': 'Softmax', 'Softsign': 'Activation', 'Softplus': 'Activation',
                     'Sum': 'Add', 'Sub': 'Subtract', 'Max': 'Maximum', 'Min': 'Minimum', 'Mul': 'Multiply', 'Concat': 'Concatenate'}

    # Define layers to skip for conversion to HLS
    skip_layers = ['Squeeze', 'Unsqueeze', 'Dropout', 'Identity', 'Flatten', 'Transpose', 'Reshape']
    # Map inputs of skipped layers
    inputs_map = {}

    passes = ['fuse_transpose_into_gemm', 'fuse_matmul_add_bias_into_gemm', 'eliminate_nop_transpose', 'fuse_consecutive_transposes']
    model = shape_inference.infer_shapes(model)  # have to infer shapes before optimizing the model
    model = optimizer.optimize(model, passes)
    model = shape_inference.infer_shapes(model)  # have to infer shapes before optimizing the model

    reader = ONNXDataReader(model)

    # Loop through layers
    layer_counter = 0
    all_inputs = [x.name for x in model.graph.input]
    all_initializers = [x.name for x in model.graph.initializer]
    input_layers = [x for x in all_inputs if x not in all_initializers]
    output_layers = [x.name for x in model.graph.output]

    for i, inp in enumerate(input_layers):
        input_layer = {}
        input_layer['name'] = inp
        input_layer['class_name'] = 'InputLayer'
        inp_shape = next((x.type.tensor_type.shape.dim for x in model.graph.input if x.name == inp), None)
        input_layer['input_shape'] = [x.dim_value for x in inp_shape]
        if len(input_layer['input_shape']) > 1:
            input_layer['input_shape'][0] = None

        input_layer['outputs'] = [inp]

        sanitize_layer_name(input_layer)
        input_layers[i] = input_layer['name']
        layer_list.append(input_layer)

    # Check for unsupported layer type
    for operation in model.graph.node:
        if operation.op_type not in supported_operations:
            raise Exception('ERROR: Unsupported operation type: {}'.format(operation.op_type))

    # Get input shape
    current_shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    print('Input shape:', current_shape)

    print('Topology:')
    for operation in model.graph.node:
        if operation.op_type == 'Flatten':
            current_shape = [current_shape[0], np.prod(current_shape[1:])]
        if operation.op_type in skip_layers:
            # Currently supported skipped layers have only one input and output
            # Skipped layers can follow each other (e.g., Dropout -> Flatten)
            input_name = inputs_map.get(operation.input[0], operation.input[0])
            output_name = operation.output[0]
            inputs_map[output_name] = input_name
            continue

        if operation.op_type in supported_operations:
            layer_counter = layer_counter + 1

        # Dictionary to fill in and append to layer_list
        layer = {}

        # Extract name for finding weights and biases
        if operation.name:
            layer['name'] = operation.name
        else:
            layer['name'] = operation.op_type + str(layer_counter)
        layer['class_name'] = operation_map.get(operation.op_type, operation.op_type)
        layer['inputs'] = [inputs_map.get(operation.input[0], operation.input[0])]
        layer['outputs'] = [x for x in operation.output]

        # Extract type of activation
        if operation.op_type in activation_operations:
            layer['activation'] = operation.op_type.lower()
            if layer_list[-1]['class_name'] != 'BatchNormalization':
                layer_list[-1]['activation'] = operation.op_type.lower()

        # Get number of inputs and outputs
        # (We take it from the weights to avoid dealing with InputLayer and Flatten details)
        if layer['class_name'] == 'Dense':
            current_shape = get_input_shape(model, operation)
            layer['n_in'] = next((x.type.tensor_type.shape.dim[-1].dim_value for x in model.graph.input if x.name == operation.input[0]), None)
            layer['n_out'] = next((x.type.tensor_type.shape.dim[-1].dim_value for x in model.graph.value_info if x.name == operation.output[0]), None)
            tran_weight = get_onnx_attribute(operation, 'transB', 0)
            reader.add_input(layer['name'], operation.input, tran_weight)

            current_shape = [current_shape[0], layer['n_out']]
        elif layer['class_name'] == 'Conv':
            current_shape = get_input_shape(model, operation)
            strides = get_onnx_attribute(operation, 'strides')
            kernel_shape = get_onnx_attribute(operation, 'kernel_shape')

            if len(current_shape) == 3:  # Conv1D
                layer['class_name'] = 'Conv1D'
                reader.add_input(layer['name'], operation.input)

                layer['in_width'] = current_shape[2]
                layer['filt_width'] = kernel_shape[0]
                layer['n_chan'] = current_shape[1]
                layer['n_filt'] = next((x.type.tensor_type.shape.dim[1].dim_value for x in model.graph.value_info if x.name == operation.output[0]), None)
                layer['stride_width'] = strides[0]
                pads = compute_pads_1d(operation, layer)

                layer['pad_left'] = pads[0]
                layer['pad_right'] = pads[1]
                if all(x == 0 for x in pads):  # No padding, i.e., 'VALID' padding
                    layer['out_width'] = int(math.ceil(float(layer['in_width'] - layer['filt_width'] + 1) / float(layer['stride'])))
                else:
                    layer['out_width'] = int(math.ceil(float(layer['in_width']) / float(layer['stride'])))

                layer['data_format'] = 'channels_first'

                current_shape = [current_shape[0], layer['n_filt'], layer['out_width']]
            elif len(current_shape) == 4:  # Conv2D
                layer['class_name'] = 'Conv2D'
                reader.add_input(layer['name'], operation.input, transpose=True, perm=[2, 3, 1, 0])

                layer['in_height'] = current_shape[2]
                layer['in_width'] = current_shape[3]
                layer['filt_height'] = kernel_shape[0]
                layer['filt_width'] = kernel_shape[1]
                layer['n_chan'] = current_shape[1]
                layer['n_filt'] = next((x.type.tensor_type.shape.dim[1].dim_value for x in model.graph.value_info if x.name == operation.output[0]), None)
                layer['stride_height'] = strides[0]
                layer['stride_width'] = strides[1]
                pads = compute_pads_2d(operation, layer)

                layer['pad_top'] = pads[0]
                layer['pad_bottom'] = pads[2]
                layer['pad_left'] = pads[1]
                layer['pad_right'] = pads[3]

                if all(x == 0 for x in pads):  # No padding, i.e., 'VALID' padding in Keras/Tensorflow
                    layer['out_width'] = int(math.ceil(float(layer['in_width'] - layer['filt_width'] + 1) / float(layer['stride_width'])))
                    layer['out_height'] = int(math.ceil(float(layer['in_height'] - layer['filt_height'] + 1) / float(layer['stride_height'])))
                else:
                    layer['out_height'] = int(math.ceil(float(layer['in_height']) / float(layer['stride_height'])))
                    layer['out_width'] = int(math.ceil(float(layer['in_width']) / float(layer['stride_width'])))

                current_shape = [current_shape[0], layer['n_filt'], layer['out_height'], layer['out_width']]
        elif layer['class_name'] == 'BatchNormalization':
            layer['epsilon'] = get_onnx_attribute(operation, 'epsilon')
            layer['momentum'] = get_onnx_attribute(operation, 'momentum')

            reader.add_input(layer['name'], operation.input)

            in_size = 1
            for dim in current_shape[1:]:
                in_size *= dim
            layer['n_in'] = in_size
            layer['n_out'] = layer['n_in']
            if len(current_shape) == 2:
                layer['n_filt'] = -1
            else:
                layer['n_filt'] = current_shape[1]
        elif layer['class_name'] in pool_operations:
            current_shape = get_input_shape(model, operation)
            info = layer['class_name'].replace('Pool', '')
            strides = get_onnx_attribute(operation, 'strides')
            kernel_shape = get_onnx_attribute(operation, 'kernel_shape')
            if len(current_shape) == 3:  # 1D
                layer['class_name'] = info + 'Pooling1D'
                layer['stride'] = strides[0]
                layer['pool_size'] = layer['y_filt'] = kernel_shape[0]
                pads = compute_pads_1d(operation, layer)
                layer['pad_left'] = pads[0]
                layer['pad_right'] = pads[1]

                if all(x == 0 for x in pads):  # No padding, i.e., 'VALID' padding
                    layer['n_out'] = int(math.ceil(float(layer['y_in'] - layer['y_filt'] + 1) / float(layer['stride'])))
                else:
                    layer['n_out'] = int(math.ceil(float(layer['y_in']) / float(layer['stride'])))

                current_shape = [current_shape[0], layer['n_filt'], layer['n_out']]
            elif len(current_shape) == 4:  # 2D
                layer['class_name'] = info + 'Pooling2D'

                layer['n_filt'] = current_shape[1]
                layer['in_height'] = current_shape[2]
                layer['in_width'] = current_shape[3]

                layer['stride_height'] = strides[0]
                layer['stride_width'] = strides[1]
                layer['pool_height'] = layer['filt_height'] = kernel_shape[0]
                layer['pool_width'] = layer['filt_width'] = kernel_shape[1]

                pads = compute_pads_2d(operation, layer)
                layer['pad_top'] = pads[0]
                layer['pad_bottom'] = pads[2]
                layer['pad_left'] = pads[1]
                layer['pad_right'] = pads[3]

                if all(x == 0 for x in pads):  # No padding, i.e., 'VALID' padding in Keras/Tensorflow
                    layer['out_width'] = int(math.ceil(float(layer['in_width'] - layer['filt_width'] + 1) / float(layer['stride_width'])))
                    layer['out_height'] = int(math.ceil(float(layer['in_height'] - layer['filt_height'] + 1) / float(layer['stride_height'])))
                else:
                    layer['out_height'] = int(math.ceil(float(layer['in_height']) / float(layer['stride_height'])))
                    layer['out_width'] = int(math.ceil(float(layer['in_width']) / float(layer['stride_width'])))

                layer['n_out'] = layer['out_height'] * layer['out_height'] * layer['n_filt']
                current_shape = [current_shape[0], layer['n_filt'], layer['out_height'], layer['out_width']]
        elif layer['class_name'] in ['ELU', 'LeakyReLU', 'ThresholdedReLU']:
            layer['activation'] = layer['class_name']
            layer['activ_param'] = get_onnx_attribute(operation, 'alpha', 0.01)
        elif layer['class_name'] == 'PReLU':
            layer['activation'] = layer['class_name']

        elif layer['class_name'] in [operation_map.get(op, op) for op in merge_operations]:
            layer['op'] = layer['class_name'].lower()
            if layer['class_name'] == 'Concatenate':
                rank = len(current_shape[1:])
                if rank > 3:
                    raise Exception('ERROR: Concatenation of tensors with rank > 3 is not yet supported.')
                layer['op'] = layer['class_name'].lower() + '{}d'.format(rank)
                layer['axis'] = get_onnx_attribute(operation, 'axis')
            else:
                layer['class_name'] = 'Merge'
            layer['inputs'] = [inputs_map.get(x, x) for x in operation.input]
            if len(layer['inputs']) > 2:
                raise Exception('ERROR: Merging more than two tensors is not yet supported.')

        sanitize_layer_name(layer)
        print('Layer name: {}, layer type: {}, current shape: {}'.format(layer['name'], layer['class_name'], current_shape))
        layer_list.append(layer)

    #################
    # Generate HLS
    #################

    print('Creating HLS model')
    hls_model = HLSModel(yamlConfig, reader, layer_list, input_layers, output_layers)
    return hls_model
