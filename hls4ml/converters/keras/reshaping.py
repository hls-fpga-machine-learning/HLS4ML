import collections.abc

from hls4ml.converters.keras_to_hls import parse_default_keras_layer
from hls4ml.converters.keras_to_hls import keras_handler


@keras_handler('ZeroPadding1D')
def parse_zeropadding1d_layer(keras_layer, input_names, input_shapes, data_reader, config):
    assert(keras_layer['class_name'] == 'ZeroPadding1D')

    layer = parse_default_keras_layer(keras_layer, input_names)

    padding = keras_layer['config']['padding']
    if isinstance(padding, int):
        layer['pad_left'] = padding
        layer['pad_right'] = padding
    elif isinstance(padding, collections.abc.Sequence):
        layer['pad_left'] = padding[0]
        layer['pad_right'] = padding[1]

    if layer['data_format'] == 'channels_first':
        output_shape = [
            input_shapes[0][0],  # Batch
            input_shapes[0][1],  # Channels
            layer['pad_left'] + input_shapes[0][2] + layer['pad_right']  # Width
        ]
        layer['out_width'] = output_shape[2]
        layer['n_chan'] = output_shape[1]
    else:
        output_shape = [
            input_shapes[0][0],  # Batch
            layer['pad_left'] + input_shapes[0][1] + layer['pad_right'],  # Width
            input_shapes[0][2]  # Channels
        ]
        layer['out_width'] = output_shape[1]
        layer['n_chan'] = output_shape[2]

    return layer, output_shape


@keras_handler('ZeroPadding2D')
def parse_zeropadding2d_layer(keras_layer, input_names, input_shapes, data_reader, config):
    assert(keras_layer['class_name'] == 'ZeroPadding2D')

    layer = parse_default_keras_layer(keras_layer, input_names)

    padding = keras_layer['config']['padding']
    if isinstance(padding, int):
        layer['pad_top'] = padding
        layer['pad_bottom'] = padding
        layer['pad_left'] = padding
        layer['pad_right'] = padding
    elif isinstance(padding, collections.abc.Sequence):
        height_pad, width_pad = padding
        if isinstance(height_pad, collections.abc.Sequence):
            layer['pad_top'] = height_pad[0]
            layer['pad_bottom'] = height_pad[1]
        else:
            layer['pad_top'] = height_pad
            layer['pad_bottom'] = height_pad
        if isinstance(width_pad, collections.abc.Sequence):
            layer['pad_left'] = width_pad[0]
            layer['pad_right'] = width_pad[1]
        else:
            layer['pad_left'] = width_pad
            layer['pad_bottom'] = width_pad

    if layer['data_format'] == 'channels_first':
        output_shape = [
            input_shapes[0][0],  # Batch
            input_shapes[0][1],  # Channels
            layer['pad_top'] + input_shapes[0][2] + layer['pad_bottom'],  # Height
            layer['pad_left'] + input_shapes[0][3] + layer['pad_right']  # Width
        ]
        layer['out_height'] = output_shape[2]
        layer['out_width'] = output_shape[3]
        layer['n_chan'] = output_shape[1]
    else:
        output_shape = [
            input_shapes[0][0],  # Batch
            layer['pad_top'] + input_shapes[0][1] + layer['pad_bottom'],  # Height
            layer['pad_left'] + input_shapes[0][2] + layer['pad_right'],  # Width
            input_shapes[0][3]  # Channels
        ]
        layer['out_height'] = output_shape[1]
        layer['out_width'] = output_shape[2]
        layer['n_chan'] = output_shape[3]

    return layer, output_shape
