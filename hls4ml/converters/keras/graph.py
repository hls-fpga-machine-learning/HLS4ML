from hls4ml.converters.keras_to_hls import parse_default_keras_layer
from hls4ml.converters.keras_to_hls import keras_handler
from hls4ml.converters.keras.core import TernaryQuantizer


@keras_handler('GarNet', 'GarNetStack')
def parse_garnet_layer(keras_layer, input_names, input_shapes, data_reader, config):
    assert(keras_layer['class_name'] in ['GarNet', 'GarNetStack'])

    if not keras_layer['config']['simplified']:
        raise Exception('HLS GarNet is compatible only with keras GarNet with simplified=True')
    if keras_layer['config']['output_activation'] is not None:
        raise Exception('HLS GarNet cannot have output activation')

    layer = parse_default_keras_layer(keras_layer, input_names)

    layer['input_format'] = keras_layer['config']['input_format']
    if layer['input_format'] != 'xn':
        raise NotImplementedError('HLS GarNet currently only implements signed inputs (input_format="xn")')

    layer['n_vertices'] = input_shapes[0][1]
    layer['collapse'] = keras_layer['config']['collapse']
    layer['mean_by_nvert'] = keras_layer['config']['mean_by_nvert']
    if keras_layer['config']['quantize_transforms']:
        layer['quantizer'] = TernaryQuantizer()

    layer['n_aggregators'] = keras_layer['config']['n_aggregators']
    layer['n_out_features'] = keras_layer['config']['n_filters']  # number of output features
    layer['n_propagate'] = keras_layer['config']['n_propagate']  # number of latent features

    if layer['class_name'] == 'GarNet':
        layer['n_in_features'] = input_shapes[0][2]
        n_out_features = layer['n_out_features']

    elif layer['class_name'] == 'GarNetStack':
        layer['n_sublayers'] = keras_layer['config']['n_sublayers']
        layer['n_in_features'] = [input_shapes[0][2]]

        for il in range(1, layer['n_sublayers']):
            layer['n_in_features'].append(layer['n_out_features'][il - 1])

        n_out_features = layer['n_out_features'][-1]

    if layer['collapse'] in ['mean', 'sum', 'max']:
        output_shape = [input_shapes[0][0], n_out_features]
    else:
        output_shape = input_shapes[0][:2] + [n_out_features]

    return layer, output_shape
