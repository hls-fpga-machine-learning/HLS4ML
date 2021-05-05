import numpy as np

from hls4ml.converters.keras_to_hls import parse_default_keras_layer
from hls4ml.converters.keras_to_hls import keras_handler

from hls4ml.model.hls_model import Quantizer
from hls4ml.model.hls_model import IntegerPrecisionType

MAXMULT =  4096

rnn_layers = ['LSTM', 'GRU']
@keras_handler(*rnn_layers)
def parse_rnn_layer(keras_layer, input_names, input_shapes, data_reader, config):
    assert(keras_layer['class_name'] in rnn_layers)

    if keras_layer['class_name'] == 'LSTM':
        div_factor = 4
    elif keras_layer['class_name'] == 'GRU':
        div_factor = 3
    else:
        div_factor = 1

    layer = parse_default_keras_layer(keras_layer, input_names)

    weights_shape = data_reader.get_weights_shape(layer['name'], 'kernel')
    recurrent_weights_shape = data_reader.get_weights_shape(layer['name'], 'recurrent_kernel')
    return_sequences_config = keras_layer['config']['return_sequences']
    layer['n_sequence'] = input_shapes[0][1]
    layer['n_sequence_out'] = layer['n_sequence'] if return_sequences_config else 1
    layer['n_in'] = weights_shape[0]
    layer['n_out'] = int(weights_shape[1]/div_factor)
    layer['n_subout']=[weights_shape[1]]
    if layer['n_in']*layer['n_out']>MAXMULT:
        n_subout = int(MAXMULT/layer['n_in'])
        n_totout = 0
        layer['n_subout'] = []
        layer['n_part'] = 0
        while n_totout < layer['n_out']:
            if n_totout + n_subout <= layer['n_out']:
                layer['n_subout'].append(n_subout)
                n_totout += n_subout
            else:
                layer['n_subout'].append(layer['n_out']-n_totout)
                n_totout += layer['n_out']-n_totout
            layer['n_part'] += 1
    layer['recurr_n_in']=recurrent_weights_shape[0]
    layer['recurr_n_out']=recurrent_weights_shape[1]
    layer['recurr_n_subout']=[recurrent_weights_shape[1]]
    layer['recurr_n_part'] = 1
    if layer['recurr_n_in']*layer['recurr_n_out']>MAXMULT:
        n_subout = int(MAXMULT/layer['recurr_n_in'])
        n_totout = 0
        layer['recurr_n_subout'] = []
        layer['recurr_n_part'] = 0
        while n_totout < layer['recurr_n_out']:
            if n_totout + n_subout <= layer['recurr_n_out']:
                layer['recurr_n_subout'].append(n_subout)
                n_totout += n_subout
            else:
                layer['recurr_n_subout'].append(layer['recurr_n_out']-n_totout)
                n_totout += layer['recurr_n_out']-n_totout
            layer['recurr_n_part'] += 1

    output_shape = [input_shapes[0][0], layer['n_sequence_out'], layer['n_out']]
    return layer, output_shape
