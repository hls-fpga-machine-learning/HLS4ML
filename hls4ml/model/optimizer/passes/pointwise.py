import numpy as np
import re

from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.hls_model import Conv1D, Conv2D, register_layer
from hls4ml.templates import templates


class PointwiseConv1D(Conv1D):
    ''' Optimized Conv1D implementation for 1x1 kernels. '''

    # Nothing to do, will pick up function and config from class name
    pass


class PointwiseConv2D(Conv2D):
    ''' Optimized Conv2D implementation for 1x1 kernels. '''

    # Nothing to do, will pick up function and config from class name
    pass


pointwise_conv1d_function_template = 'nnet::pointwise_conv_1d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'
pointwise_conv2d_function_template = 'nnet::pointwise_conv_2d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'

sepconv1d_include_list = ['nnet_utils/nnet_conv1d.h', 'nnet_utils/nnet_sepconv1d_stream.h']
sepconv2d_include_list = ['nnet_utils/nnet_conv2d.h', 'nnet_utils/nnet_sepconv2d_stream.h']

# Register the layer types to the layer map
register_layer('PointwiseConv1D', PointwiseConv1D)
register_layer('PointwiseConv2D', PointwiseConv2D)

# Register the templates for config and function
templates.get_backend('Vivado').register_templates(
    'PointwiseConv1D',
    pointwise_conv1d_function_template,
    templates.get_backend('Vivado').get_config_template('Conv1D'),
    sepconv1d_include_list
)

templates.get_backend('Vivado').register_templates(
    'PointwiseConv2D',
    pointwise_conv2d_function_template,
    templates.get_backend('Vivado').get_config_template('Conv2D'),
    sepconv2d_include_list
)


class OptimizePointwiseConv(OptimizerPass):
    def match(self, node):
        return node.__class__.__name__ in ['Conv1D', 'Conv2D'] and \
            node.get_attr('filt_height', 1) == 1 and \
            node.get_attr('filt_width') == 1

    def transform(self, model, node):
        dim = node.__class__.__name__[-2:]  # '1D' or '2D'
        pw_node = model.make_node('PointwiseConv' + dim, node.name, node.attributes.copy(), node.inputs.copy())
        model.replace_node(node, pw_node)

        return True
