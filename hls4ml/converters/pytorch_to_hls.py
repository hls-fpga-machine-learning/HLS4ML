from __future__ import print_function
import numpy as np
import os
import yaml
import sys
import torch
import pickle
import re

from hls4ml.model import HLSModel


class PyTorchDataReader:
    def __init__(self, config):
        self.config = config

        if not torch.cuda.is_available():
            self.torch_model = torch.load(config['PytorchModel'], map_location=lambda storage, loc: storage)
        else:
            self.torch_model = torch.load(config['PytorchModel'])

        self.state_dict = self.torch_model.state_dict()

    def get_weights_data(self, layer_name, var_name):
        if var_name == 'kernel':
            var_name = 'weight'
        data = None
        if var_name in ['weight', 'bias']:
            data = self.state_dict[layer_name + '.' + var_name].numpy().transpose()

        return data


def pytorch_to_hls(yamlConfig):

    ######################
    # Do translation
    ######################

    print('Interpreting Model')
    reader = PyTorchDataReader(yamlConfig)

    core_layers = ['Linear']
    skip_layers = ['Dropout', 'Flatten']
    activation_layers = ['ReLU', 'Sigmoid', 'Tanh', 'SELU', 'LeakyReLU', 'Softmax', 'Softplus', 'Softsign']
    supported_layers = core_layers + skip_layers + activation_layers

    # This is a list of dictionaries to hold all the layer info we need to generate HLS
    layer_list = []

    # Loop through layers
    print('Topology:')
    modelstr = repr(reader.torch_model).split('\n')
    for pytorch_layer in modelstr:
        layer_match = re.match(r'\((\d)\): (\w+)\((.*)\)', pytorch_layer.strip())
        if layer_match is None:
            continue

        layer_idx = layer_match.group(1)
        layer_type = layer_match.group(2)
        layer_spec = layer_match.group(3)

        # #Dictionary to fill in and append to layer_list
        layer = {}

        #layer_type = matchname.group(1)
        if layer_type not in supported_layers:
            raise Exception('Unsupported layer {}'.format(layer_type))

        if layer_type == 'Linear':
            layer['class_name'] = 'Dense'
            layer['name'] = layer_idx

            dense_spec = re.match(r'in_features=(\d+), out_features=(\d+).*', layer_spec)
            if dense_spec is None:
                raise Exception('Unable to interpret Linear layer ({})'.format(layer_spec))

            # #Get number of inputs and outputs
            layer['n_in'] = int(dense_spec.group(1))
            layer['n_out'] = int(dense_spec.group(2))

            current_shape = [layer['n_in'], layer['n_out']]
            print('Layer index: {}, layer type: {}, current shape: {}'.format(layer['name'], layer['class_name'], current_shape))
        elif layer_type in activation_layers:
            layer['activation'] = layer_type.lower()
            if layer['activation'] == 'Softmax':
                layer['class_name'] = 'Softmax'
            else:
                layer['class_name'] = 'Activation'
            layer['name'] = layer['activation'] + '_' + str(layer_idx)

        layer_list.append(layer)

    input_layer = {}
    input_layer['name'] = 'input1'
    input_layer['class_name'] = 'InputLayer'
    input_layer['input_shape'] = [layer_list[0]['n_in']]
    layer_list.insert(0, input_layer)

    #################
    # Generate HLS
    #################

    reader = PyTorchDataReader(yamlConfig)
    print('Creating HLS model')
    hls_model = HLSModel(yamlConfig, reader, layer_list)
    return hls_model
