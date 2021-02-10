from hls4ml.converters.keras_to_hls import parse_default_keras_layer
from hls4ml.converters.keras_to_hls import keras_handler

merge_layers = ['Add', 'Subtract', 'Multiply', 'Average', 'Maximum', 'Minimum', 'Concatenate', 'Dot']


@keras_handler(*merge_layers)
def parse_merge_layer(keras_layer, input_names, input_shapes, data_reader, config):
    assert(keras_layer['class_name'] in merge_layers)

    layer = parse_default_keras_layer(keras_layer, input_names)

    layer['op'] = layer['class_name'].lower()

    if layer['class_name'] == 'Concatenate':
        rank = len(input_shapes[0][1:])
        if rank > 3:
            raise Exception('ERROR: Concatenation of tensors with rank > 3 is not yet supported.')
        layer['op'] = layer['class_name'].lower() + '{}d'.format(rank)
        layer['axis'] = keras_layer['config']['axis']
        # TODO handle output shape
    elif layer['class_name'] == 'Dot':
        rank = len(input_shapes[0][1:])
        if rank > 1:
            raise Exception('ERROR: Dot of tensors with rank > 1 is not yet supported.')
        layer['op'] = layer['class_name'].lower() + '{}d'.format(rank)
    else:
        layer['class_name'] = 'Merge'
    if len(layer['inputs']) > 2:
        raise Exception('ERROR: Merging more than two tensors is not yet supported.')

    return layer, input_shapes[0]
