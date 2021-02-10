from hls4ml.converters.keras_to_hls import parse_default_keras_layer
from hls4ml.converters.keras_to_hls import keras_handler

from hls4ml.converters.keras.core import parse_dense_layer
from hls4ml.converters.keras.core import parse_batchnorm_layer
from hls4ml.converters.keras.convolution import parse_conv1d_layer
from hls4ml.converters.keras.convolution import parse_conv2d_layer
from hls4ml.converters.keras.qkeras import *

import tensorflow as tf


@keras_handler('QDense')
def parse_qdense_layer(keras_layer, input_names, input_shapes, data_reader, config):

    layer, output_shape = parse_dense_layer(keras_layer, input_names, input_shapes, data_reader, config)

    layer['weight_quantizer'] = get_quantizer_from_config(keras_layer, 'kernel')
    if keras_layer['config']['bias_quantizer'] is not None:
        layer['bias_quantizer'] = get_quantizer_from_config(keras_layer, 'bias')
    else:
        layer['bias_quantizer'] = None

    return layer, output_shape


@keras_handler('QConv1D', 'QConv2D')
def parse_qconv_layer(keras_layer, input_names, input_shapes, data_reader, config):
    assert('QConv' in keras_layer['class_name'])

    if int(keras_layer['class_name'][-2]) == 1:
        layer, output_shape = parse_conv1d_layer(keras_layer, input_names, input_shapes, data_reader, config)
    elif int(keras_layer['class_name'][-2]) == 2:
        layer, output_shape = parse_conv2d_layer(keras_layer, input_names, input_shapes, data_reader, config)

    layer['weight_quantizer'] = get_quantizer_from_config(keras_layer, 'kernel')
    if keras_layer['config']['bias_quantizer'] is not None:
        layer['bias_quantizer'] = get_quantizer_from_config(keras_layer, 'bias')
    else:
        layer['bias_quantizer'] = None

    return layer, output_shape


@keras_handler('QActivation')
def parse_qactivation_layer(keras_layer, input_names, input_shapes, data_reader, config):
    assert(keras_layer['class_name'] == 'QActivation')
    supported_activations = ['quantized_relu', 'quantized_tanh', 'binary_tanh', 'ternary_tanh', 'quantized_bits', 'binary', 'ternary']

    layer = parse_default_keras_layer(keras_layer, input_names)

    activation_config = keras_layer['config']['activation']
    if isinstance(activation_config, str):
        quantizer_obj = get_quantizer(activation_config)

    if isinstance(activation_config, str):
        quantizer_obj = get_quantizer(activation_config)
        activation_config = {}
        # some activations are classes
        if hasattr(quantizer_obj, 'get_config'):
            print("Name: " + quantizer_obj.__class__.__name__)
            activation_config['class_name'] = quantizer_obj.__class__.__name__
            if activation_config['class_name'] == 'ternary' or activation_config['class_name'] == 'binary':
                activation_config['class_name'] += '_tanh'
            activation_config['config'] = quantizer_obj.get_config()
        # some activation quantizers are just functions with no config
        else:
            activation_config['config'] = {}
            if 'binary' in quantizer_obj.__name__:
                activation_config['class_name'] = 'binary_tanh'
                activation_config['config']['bits'] = 1
                activation_config['config']['integer'] = 1
            elif 'ternary' in quantizer_obj.__name__:
                activation_config['class_name'] = 'ternary_tanh'
                activation_config['config']['bits'] = 2
                activation_config['config']['integer'] = 2
            else:
                activation_config['class_name'] = 'unknown'

    if activation_config['class_name'] not in supported_activations:
        raise Exception('Unsupported QKeras activation: {}'.format(activation_config['class_name']))

    layer['class_name'] = 'Activation'
    if activation_config['class_name'] == 'quantized_bits':
        activation_config['class_name'] = 'linear'
    layer['activation'] = activation_config['class_name'].replace('quantized_', '')
    return layer, [shape for shape in input_shapes[0]]


@keras_handler('QBatchNormalization')
def parse_qbatchnorm_layer(keras_layer, input_names, input_shapes, data_reader, config):

    layer, output_shape = parse_batchnorm_layer(keras_layer, input_names, input_shapes, data_reader, config)

    layer['mean_quantizer'] = get_quantizer_from_config(keras_layer, 'mean')
    layer['variance_quantizer'] = get_quantizer_from_config(keras_layer, 'variance')
    layer['beta_quantizer'] = get_quantizer_from_config(keras_layer, 'beta')
    layer['gamma_quantizer'] = get_quantizer_from_config(keras_layer, 'gamma')

    return layer, output_shape
