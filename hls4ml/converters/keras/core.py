import numpy as np

from hls4ml.converters.keras_to_hls import parse_default_keras_layer
from hls4ml.converters.keras_to_hls import keras_handler

from hls4ml.model.hls_model import Quantizer
from hls4ml.model.hls_model import IntegerPrecisionType


@keras_handler('InputLayer')
def parse_input_layer(keras_layer, input_names, input_shapes, data_reader, config):
    assert(keras_layer['class_name'] == 'InputLayer')

    layer = parse_default_keras_layer(keras_layer, input_names)

    layer['input_shape'] = keras_layer['config']['batch_input_shape'][1:]
    if keras_layer['config']['dtype'] == 'int32':
        layer['type_name'] = 'integer_input_t'
        layer['precision'] = IntegerPrecisionType(width=32)
    output_shape = keras_layer['config']['batch_input_shape']

    return layer, output_shape


class BinaryQuantizer(Quantizer):
    def __init__(self, bits=2):
        if bits == 1:
            hls_type = IntegerPrecisionType(width=1, signed=False)
        elif bits == 2:
            hls_type = IntegerPrecisionType(width=2)
        else:
            raise Exception('BinaryQuantizer suppots 1 or 2 bits, but called with bits={}'.format(bits))
        super(BinaryQuantizer, self).__init__(bits, hls_type)

    def __call__(self, data):
        zeros = np.zeros_like(data)
        ones = np.ones_like(data)
        quant_data = data
        if self.bits == 1:
            quant_data = np.where(data > 0, ones, zeros).astype('int')
        if self.bits == 2:
            quant_data = np.where(data > 0, ones, -ones)
        return quant_data


class TernaryQuantizer(Quantizer):
    def __init__(self):
        super(TernaryQuantizer, self).__init__(2, IntegerPrecisionType(width=2))

    def __call__(self, data):
        zeros = np.zeros_like(data)
        ones = np.ones_like(data)
        return np.where(data > 0.5, ones, np.where(data <= -0.5, -ones, zeros))


dense_layers = ['Dense', 'BinaryDense', 'TernaryDense']


@keras_handler(*dense_layers)
def parse_dense_layer(keras_layer, input_names, input_shapes, data_reader, config):
    assert('Dense' in keras_layer['class_name'])

    layer = parse_default_keras_layer(keras_layer, input_names)

    weights_shape = data_reader.get_weights_shape(layer['name'], 'kernel')
    layer['n_in'] = weights_shape[0]
    layer['n_out'] = weights_shape[1]
    if 'Binary' in layer['class_name']:
        layer['weight_quantizer'] = BinaryQuantizer(bits=2)
        layer['bias_quantizer'] = BinaryQuantizer(bits=2)
    elif 'Ternary' in layer['class_name']:
        layer['weight_quantizer'] = TernaryQuantizer()
        layer['bias_quantizer'] = TernaryQuantizer()
    else:
        layer['weight_quantizer'] = None
        layer['bias_quantizer'] = None
    output_shape = [input_shapes[0][0], layer['n_out']]

    return layer, output_shape


activation_layers = ['Activation', 'LeakyReLU', 'ThresholdedReLU', 'ELU', 'PReLU', 'Softmax', 'ReLU']


@keras_handler(*activation_layers)
def parse_activation_layer(keras_layer, input_names, input_shapes, data_reader, config):
    assert(keras_layer['class_name'] in activation_layers)

    layer = parse_default_keras_layer(keras_layer, input_names)

    if layer['class_name'] != 'Activation':
        layer['activation'] = layer['class_name']
    if layer['class_name'] == 'LeakyReLU':
        layer['activ_param'] = keras_layer["config"].get('alpha', 0.3)
    elif layer['class_name'] == 'ThresholdedReLU':
        layer['activ_param'] = keras_layer["config"].get('theta', 1.)
    elif layer['class_name'] == 'ELU':
        layer['activ_param'] = keras_layer["config"].get('alpha', 1.)
    elif layer['class_name'] == 'ReLU':
        layer['class_name'] = 'Activation'

    if layer['class_name'] == 'Activation' and layer['activation'] == 'softmax':
        layer['class_name'] = 'Softmax'
    if layer['class_name'] == 'ReLU':
        layer['class_name'] = 'Activation'

    return layer, [shape for shape in input_shapes[0]]


@keras_handler('BatchNormalization')
def parse_batchnorm_layer(keras_layer, input_names, input_shapes, data_reader, config):
    assert('BatchNormalization' in keras_layer['class_name'])

    layer = parse_default_keras_layer(keras_layer, input_names)

    in_size = 1
    for dim in input_shapes[0][1:]:
        in_size *= dim
    layer['n_in'] = in_size
    layer['n_out'] = layer['n_in']
    if len(input_shapes[0]) == 2:
        layer['n_filt'] = -1
    elif len(input_shapes[0]) == 3:
        layer['n_filt'] = input_shapes[0][2]
    elif len(input_shapes[0]) == 4:
        layer['n_filt'] = input_shapes[0][3]

    return layer, [shape for shape in input_shapes[0]]
