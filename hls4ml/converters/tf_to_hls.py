from __future__ import print_function
import numpy as np
import h5py
import json
import math
import os
import tensorflow as tf
from tensorflow.python.framework import tensor_util

from hls4ml.model import HLSModel

MAXMULT = 4096


class TFDataReader:
    def __init__(self, graph):
        self.graph = graph
        self.const_ops = [c for c in self.graph.get_operations() if c.type == 'Const']

    def get_weights_data(self, layer_name, var_name):
        tf_op = self.graph.get_operation_by_name(layer_name)
        data = None
        if tf_op is not None:
            if tf_op.type == 'MatMul' and var_name == 'kernel':  # MatMul is mapped to Dense, but there is no bias
                w_tensor = tf_op.inputs[1]
                data = self.read_variable_data(w_tensor)

            if tf_op.type == 'Conv2D' and var_name == 'kernel':
                w_tensor = tf_op.inputs[1]
                data = self.read_variable_data(w_tensor)

            if tf_op.type == 'BiasAdd':
                b_tensor = tf_op.inputs[1]
                data = self.read_variable_data(b_tensor)

            if tf_op.type == 'FusedBatchNorm':
                bn_weighs_map = {'gamma': 1, 'beta': 2, 'moving_mean': 3, 'moving_variance': 4}
                w_idx = bn_weighs_map[var_name]
                w_tensor = tf_op.inputs[w_idx]
                data = self.read_variable_data(w_tensor)

        return data

    def read_variable_data(self, tensor):
        parent_op = tensor.op
        while parent_op.type != 'Const':
            tensor = parent_op.inputs[0]
            parent_op = tensor.op

        data = tensor_util.MakeNdarray(parent_op.node_def.attr['value'].tensor)

        return data


def _parse_data_format(data_format):
    if data_format == 'NCHW':
        format_str = 'channels_first'
        c_idx = 1
        h_idx = 2
        w_idx = 3
    else:
        format_str = 'channels_last'
        h_idx = 1
        w_idx = 2
        c_idx = 3

    return format_str, c_idx, h_idx, w_idx


def _compute_pads_2d(layer, in_height, in_width):
    if layer['padding'] == 'same':
        # Height
        layer['out_height'] = int(math.ceil(float(in_height) / float(layer['stride_height'])))
        if (in_height % layer['stride_height'] == 0):
            pad_along_height = max(layer['filt_height'] - layer['stride_height'], 0)
        else:
            pad_along_height = max(layer['filt_height'] - (in_height % layer['stride_height']), 0)
        layer['pad_top'] = pad_along_height // 2
        layer['pad_bottom'] = pad_along_height - layer['pad_top']
        # Width
        layer['out_width'] = int(math.ceil(float(in_width) / float(layer['stride_width'])))
        if (in_width % layer['stride_width'] == 0):
            pad_along_width = max(layer['filt_width'] - layer['stride_width'], 0)
        else:
            pad_along_width = max(layer['filt_width'] - (in_width % layer['stride_width']), 0)
        layer['pad_left'] = pad_along_width // 2
        layer['pad_right'] = pad_along_width - layer['pad_left']
    elif layer['padding'] == 'valid':
        layer['out_width'] = int(math.ceil(float(in_width - layer['filt_width'] + 1) / float(layer['stride_width'])))
        layer['out_height'] = int(math.ceil(float(in_height - layer['filt_height'] + 1) / float(layer['stride_height'])))
        layer['pad_top'] = 0
        layer['pad_bottom'] = 0
        layer['pad_left'] = 0
        layer['pad_right'] = 0
    elif layer['padding'] == 'explicit':
        #paddings = tf_op.get_attr('explicit_paddings')
        raise NotImplementedError('Explicit padding is not supported yet.')


def _find_graph_outputs(graph):
    input_list = []
    output_list = []

    for tf_op in graph.get_operations():
        input_list.extend(tf_op.inputs)
        if tf_op.type != 'FusedBatchNorm':
            output_list.extend(tf_op.outputs)

    return [o.name.rsplit(':', 1)[0] for o in list(set(output_list) - set(input_list))]


def _parse_tensor_names(tensors):
    if not isinstance(tensors, list):
        tensors = [tensors]

    tensor_names = []
    for t in tensors:
        tensor_names.append(t.name.rsplit(':', 1)[0])

    return tensor_names


def tf_to_hls(yamlConfig):

    ######################
    # Do translation
    ######################

    # This is a list of dictionaries to hold all the layer info we need to generate HLS
    layer_list = []

    if not os.path.exists(yamlConfig['TensorFlowModel']):
        raise Exception('The specified file does not exist: {}'.format(yamlConfig['TensorFlowModel']))

    graph_def = None
    graph = None

    # Extract model architecture from pb
    try:
        with tf.io.gfile.GFile(yamlConfig['TensorFlowModel'], "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
    except BaseException as e:
        raise Exception('Error loading the graph definition: {}'.format(str(e)))

    try:
        assert graph_def is not None
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def,
                input_map=None,
                return_elements=None,
                name='',
                producer_op_list=None
            )
    except BaseException as e:
        raise Exception('Error importing the graph: {}'.format(str(e)))

    # Define supported operations
    array_ops = ['ConcatV2', 'StridedSlice', 'Transpose']
    core_ops = ['Const', 'Identity', 'Placeholder']
    image_ops = ['ResizeNearestNeighbor']
    math_ops = ['Add', 'MatMul', 'Mul', 'Sigmoid']
    nn_ops = ['AvgPool', 'BiasAdd', 'Conv2D', 'Elu', 'FusedBatchNorm', 'MaxPool', 'Relu', 'Selu', 'Softmax']
    supported_ops = array_ops + core_ops + image_ops + math_ops + nn_ops

    input_layers = []
    output_layers = _find_graph_outputs(graph)

    # Get input shape and check for unsupported layer type
    output_shape = None
    for tf_op in graph.get_operations():
        if tf_op.type not in supported_ops:
            raise Exception('ERROR: Unsupported layer type: {}'.format(tf_op.type))

    print('Topology:')
    for tf_op in graph.get_operations():
        handled = False

        layer = {}
        layer['name'] = tf_op.name

        if tf_op.type == 'Placeholder':
            if len(tf_op.inputs) == 0:  # Input
                output_shape = tf_op.outputs[0].shape.as_list()
                layer['class_name'] = 'InputLayer'
                layer['input_shape'] = output_shape[1:]
                #layer['outputs'] = [tf_op.outputs[0].name for o in tf_op.outputs]
                layer['outputs'] = _parse_tensor_names(tf_op.outputs)
                input_layers.append(layer['name'])
                handled = True

        elif tf_op.type == 'Const' or tf_op.type == 'Identity':
            # Nothing to do here, TFDataReader handles these
            handled = True
            continue

        elif tf_op.type == 'MatMul':
            input_shape = tf_op.inputs[0].shape.as_list()
            output_shape = tf_op.outputs[0].shape.as_list()
            layer['class_name'] = 'Dense'
            layer['inputs'] = _parse_tensor_names(tf_op.inputs[0])
            layer['outputs'] = _parse_tensor_names(tf_op.outputs[0])
            layer['n_in'] = input_shape[-1]
            layer['n_out'] = output_shape[-1]
            handled = True

        elif tf_op.type == 'BiasAdd':
            input_shape = tf_op.inputs[0].shape.as_list()
            output_shape = tf_op.outputs[0].shape.as_list()
            layer['class_name'] = 'BiasAdd'
            layer['op'] = 'Add'
            layer['inputs'] = _parse_tensor_names(tf_op.inputs[0])
            layer['outputs'] = _parse_tensor_names(tf_op.outputs[0])
            handled = True

        elif tf_op.type in ['Elu', 'Relu', 'Selu', 'Sigmoid', 'Softmax']:
            output_shape = tf_op.outputs[0].shape.as_list()
            if tf_op.type == 'Softmax':
                layer['class_name'] = 'Softmax'
            else:
                layer['class_name'] = 'Activation'
            layer['activation'] = tf_op.type
            layer['inputs'] = _parse_tensor_names(tf_op.inputs[0])
            layer['outputs'] = _parse_tensor_names(tf_op.outputs[0])
            handled = True

        elif tf_op.type == 'Conv2D':
            input_shape = tf_op.inputs[0].shape.as_list()
            weights_shape = tf_op.inputs[1].shape.as_list()
            output_shape = tf_op.outputs[0].shape.as_list()
            layer['data_format'], c_idx, h_idx, w_idx = _parse_data_format(tf_op.get_attr('data_format').decode())
            dilations = tf_op.get_attr('dilations')
            strides = tf_op.get_attr('strides')

            layer['class_name'] = 'Conv2D'
            layer['inputs'] = _parse_tensor_names(tf_op.inputs[0])
            layer['outputs'] = _parse_tensor_names(tf_op.outputs[0])

            layer['n_chan'] = input_shape[c_idx]
            layer['in_height'] = input_shape[h_idx]
            layer['in_width'] = input_shape[w_idx]

            # weights_shape = (filter_height, filter_width, n_channels, n_filters)
            layer['filt_height'] = weights_shape[0]
            layer['filt_width'] = weights_shape[1]
            layer['n_chan'] = weights_shape[2]
            layer['n_filt'] = weights_shape[3]

            layer['stride_height'] = strides[h_idx]
            layer['stride_width'] = strides[w_idx]
            layer['dilation_height'] = dilations[h_idx]
            layer['dilation_width'] = dilations[w_idx]

            layer['padding'] = tf_op.get_attr('padding').decode().lower()
            in_height = input_shape[h_idx]
            in_width = input_shape[w_idx]
            _compute_pads_2d(layer, in_height, in_width)

            handled = True

        elif tf_op.type == 'MaxPool':
            input_shape = tf_op.inputs[0].shape.as_list()
            output_shape = tf_op.outputs[0].shape.as_list()
            layer['data_format'], c_idx, h_idx, w_idx = _parse_data_format(tf_op.get_attr('data_format').decode())
            strides = tf_op.get_attr('strides')
            kernel_size = tf_op.get_attr('ksize')

            layer['class_name'] = 'MaxPooling2D'
            layer['inputs'] = _parse_tensor_names(tf_op.inputs[0])
            layer['outputs'] = _parse_tensor_names(tf_op.outputs[0])

            layer['padding'] = tf_op.get_attr('padding').decode().lower()

            layer['in_height'] = input_shape[h_idx]
            layer['in_width'] = input_shape[w_idx]
            layer['n_filt'] = input_shape[c_idx]

            layer['stride_height'] = strides[h_idx]
            layer['stride_width'] = strides[w_idx]
            layer['filt_height'] = layer['pool_height'] = kernel_size[h_idx]
            layer['filt_width'] = layer['pool_width'] = kernel_size[w_idx]

            layer['padding'] = tf_op.get_attr('padding').decode().lower()
            in_height = input_shape[h_idx]
            in_width = input_shape[w_idx]
            _compute_pads_2d(layer, in_height, in_width)

            handled = True

        elif tf_op.type == 'FusedBatchNorm':
            input_shape = tf_op.inputs[0].shape.as_list()
            output_shape = tf_op.outputs[0].shape.as_list()

            layer['class_name'] = 'BatchNormalization'
            layer['inputs'] = _parse_tensor_names(tf_op.inputs[0])
            layer['outputs'] = _parse_tensor_names(tf_op.outputs[0])
            layer['data_format'], c_idx, h_idx, w_idx = _parse_data_format(tf_op.get_attr('data_format').decode())
            layer['n_in'] = np.prod(input_shape[1:])
            layer['epsilon'] = tf_op.get_attr('epsilon')

            if len(input_shape) < 4:
                layer['n_filt'] = -1
            else:
                layer['n_filt'] = input_shape[c_idx]

            handled = True

        elif tf_op.type == 'ConcatV2':
            layer['class_name'] = 'Concatenate'
            layer['inputs'] = _parse_tensor_names(tf_op.inputs[:-1])
            layer['outputs'] = _parse_tensor_names(tf_op.outputs[0])
            output_shape = tf_op.outputs[0].shape.as_list()

            rank = tf_op.get_attr('N')
            if rank != 2:
                raise Exception('Unsupported number of inputs in Concat operation')

            layer['op'] = layer['class_name'].lower() + '{}d'.format(rank)
            layer['axis'] = tf_op.inputs[2].op.node_def.attr['value'].tensor.int_val[0]  # Urgh!

            handled = True

        elif tf_op.type in ['Add', 'Mul']:
            layer['class_name'] = 'Merge'
            layer['inputs'] = _parse_tensor_names(list(tf_op.inputs))
            layer['outputs'] = _parse_tensor_names(tf_op.outputs[0])
            output_shape = tf_op.outputs[0].shape.as_list()

            layer['op'] = tf_op.type.lower()
            if layer['op'] == 'mul':
                layer['op'] = 'multiply'

            handled = True

        elif tf_op.type == 'Transpose':
            layer['class_name'] = 'Transpose'
            layer['inputs'] = _parse_tensor_names(tf_op.inputs[0])
            layer['outputs'] = _parse_tensor_names(tf_op.outputs[0])
            layer['perm'] = tensor_util.MakeNdarray(tf_op.inputs[1].op.node_def.attr['value'].tensor).tolist()
            output_shape = tf_op.outputs[0].shape.as_list()

            handled = True

        elif tf_op.type == 'ResizeNearestNeighbor':
            layer['class_name'] = 'Resize'
            layer['algorithm'] = 'nearest'
            layer['inputs'] = _parse_tensor_names(tf_op.inputs[0])
            layer['outputs'] = _parse_tensor_names(tf_op.outputs[0])

            input_shape = tf_op.inputs[0].shape.as_list()  # (B, H, W, C)
            output_shape = tf_op.outputs[0].shape.as_list()
            layer['in_height'] = input_shape[1]
            layer['in_width'] = input_shape[2]
            layer['n_chan'] = input_shape[3]
            layer['out_height'] = output_shape[1]
            layer['out_width'] = output_shape[2]

            # Check for currently unsupported operations
            align_corners = tf_op.get_attr('align_corners')
            if align_corners:
                raise NotImplementedError('Property "align_corners=True" is not supported.')
            half_pixel_centers = tf_op.get_attr('align_corners')
            if half_pixel_centers:
                raise NotImplementedError('Property "half_pixel_centers=True" is not supported.')

            handled = True

        if not handled:
            raise Exception('Unable to parse operation: {} - {}'.format(tf_op.type, tf_op.name))

        print('Layer name: {}, layer type: {}, current shape: {}'.format(layer['name'], layer['class_name'], output_shape))
        layer_list.append(layer)

    #################
    # Generate HLS
    #################

    reader = TFDataReader(graph)
    print('Creating HLS model')
    hls_model = HLSModel(yamlConfig, reader, layer_list, input_layers, output_layers)
    return hls_model
