from __future__ import print_function
import numpy as np
import h5py
import os
import tarfile
import json
import argparse
import yaml
import sys
from shutil import copyfile
import math

MAXMULT = 4096000

filedir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(filedir, "..", "hls-writer"))
from hls_writer import parse_config, print_array_to_cpp, hls_writer

def find_kernel_in_h5(name):
    if 'kernel' in name:
        return name

def find_bias_in_h5(name):
    if 'bias' in name:
        return name

def find_beta_in_h5(name):
    if 'beta' in name:
        return name

def find_moving_mean_in_h5(name):
    if 'moving_mean' in name:
        return name

def find_moving_variance_in_h5(name):
    if 'moving_variance' in name:
        return name

def find_gamma_in_h5(name):
    if 'gamma' in name:
        return name

############################################################################################
## M A I N
############################################################################################
def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-c", action='store', dest='config',
                        help="Configuration file.")
    args = parser.parse_args()
    if not args.config: parser.error('A configuration file needs to be specified.')

    configDir  = os.path.abspath(os.path.dirname(args.config))
    yamlConfig = parse_config(args.config)
    if not os.path.isabs(yamlConfig['OutputDir']):
        yamlConfig['OutputDir'] = os.path.join(configDir, yamlConfig['OutputDir'])
    if not os.path.isabs(yamlConfig['KerasH5']):
        yamlConfig['KerasH5'] = os.path.join(configDir, yamlConfig['KerasH5'])
    if not os.path.isabs(yamlConfig['KerasJson']):
        yamlConfig['KerasJson'] = os.path.join(configDir, yamlConfig['KerasJson'])

    if not (yamlConfig["IOType"] == "io_parallel" or yamlConfig["IOType"] == "io_serial"): 
        raise Exception('ERROR: Invalid IO type')

    ######################
    ##  Do translation
    ######################
    if not os.path.isdir("{}/firmware/weights".format(yamlConfig['OutputDir'])):
        os.makedirs("{}/firmware/weights".format(yamlConfig['OutputDir']))

    h5File = h5py.File( yamlConfig['KerasH5'], 'r' )

    #This is a list of dictionaries to hold all the layer info we need to generate HLS
    layer_list = []

    #Extract model architecture from json
    with open( yamlConfig['KerasJson'] ) as json_file:
        model_arch = json.load(json_file)
    #print(model_arch)

    #Define supported laers
    supported_layers = ['InputLayer','Dropout', 'Flatten', 'Dense', 'Conv1D', 'Conv2D', 'BatchNormalization', 'MaxPooling1D', 'MaxPooling2D', 'AveragePooling1D', 'AveragePooling2D']
    activation_layers = ['Activation', 'LeakyReLU', 'ThresholdedReLU', 'ELU', 'PReLU']

    #Define layers to skip for conversion to HLS
    skip_layers = ['InputLayer','Dropout', 'Flatten'] 

    #Loop through layers
    layer_counter = 0
    input_layer = {}

    layer_config = None
    if model_arch['class_name'] == 'Sequential':
        print('Interpreting Sequential')
        layer_config = model_arch["config"]
    elif model_arch['class_name'] == 'Model':
        print('Interpreting Model')
        layer_config = model_arch["config"]["layers"]

    # Get input shape and check for unsupported layer type
    current_shape = None
    for keras_layer in layer_config:
        if keras_layer["class_name"] not in supported_layers + activation_layers:
            raise Exception('ERROR: Unsupported layer type: {}'.format(keras_layer["class_name"]))
        if 'batch_input_shape' in keras_layer['config']:
            current_shape = keras_layer['config']['batch_input_shape'] # [None, 100, 7]    
    print('Input shape:', current_shape)

    # Set some variables to make the routine after a bit smoother
    is_conv2d = False
    is_dense = False
    for keras_layer in layer_config:
     if keras_layer["class_name"]=='Conv2D':
      is_conv2d = True
      break
     if keras_layer["class_name"]=='Dense':
      is_dense = True
      break
	        
    print('Topology:')
    for il,keras_layer in enumerate(layer_config):
        if keras_layer["class_name"] is 'Flatten':
            current_shape = [current_shape[0], np.prod(current_shape[1:])]
        if keras_layer["class_name"] in skip_layers:
            continue 

        if keras_layer["class_name"] in supported_layers + activation_layers:
            layer_counter = layer_counter + 1

        #Dictionary to fill in and append to layer_list
        layer = {}

        #Extract name for finding weights and biases
        layer['name']=keras_layer['config']['name']
        layer['class_name']=keras_layer['class_name']

        #Extract type of activation and number of nodes
        for config,config_value in keras_layer["config"].items():
            if(config=="activation"):
                layer['activation']=config_value
            if(config=="epsilon"):
                layer['epsilon']=config_value	
            #if(config=="units"):
                #print("PARSED NUM OF NODES",config_value)

        
        #Translate weights and biases from h5 file
        if layer['class_name'] != 'BatchNormalization' and layer['class_name'] not in activation_layers and 'Pooling' not in layer['class_name']:
            found_weights = h5File[layer['name']].visit(find_kernel_in_h5)
            weights = h5File['/{}/{}'.format(layer['name'],found_weights)][()]
            found_bias = h5File[layer['name']].visit(find_bias_in_h5)
            biases = h5File['/{}/{}'.format(layer['name'],found_bias)][()]
            cur_n_zeros = print_array_to_cpp("w{}".format(layer_counter), weights, yamlConfig['OutputDir'])
            print_array_to_cpp("b{}".format(layer_counter), biases, yamlConfig['OutputDir'])
            layer['weights_n_zeros'] = cur_n_zeros
        elif layer['class_name'] == 'BatchNormalization':
            cur_n_zeros = []
            layer['weights_n_zeros'] = cur_n_zeros 
            found_beta = h5File[layer['name']].visit(find_beta_in_h5)
            beta = h5File['/{}/{}'.format(layer['name'],found_beta)][()]
            print_array_to_cpp("beta{}".format(layer_counter), beta, yamlConfig['OutputDir'])
            found_mean = h5File[layer['name']].visit(find_moving_mean_in_h5)
            mean = h5File['/{}/{}'.format(layer['name'],found_mean)][()]
            print_array_to_cpp("mean{}".format(layer_counter), mean, yamlConfig['OutputDir'])
            found_gamma = h5File[layer['name']].visit(find_gamma_in_h5)
            gamma = h5File['/{}/{}'.format(layer['name'],found_gamma)][()]
            found_var = h5File[layer['name']].visit(find_moving_variance_in_h5)
            var = h5File['/{}/{}'.format(layer['name'],found_var)][()]
            var = var + layer['epsilon']
            scale = gamma/np.sqrt(var)
            print_array_to_cpp("scale{}".format(layer_counter), scale, yamlConfig['OutputDir'])
        
        # Skip activation layers if possible
        skip_layer = False
        # Default one layer call
        layer['n_part'] = 1
        #Get number of inputs and outputs
        #(We take it from the weights to avoid dealing with InputLayer and Flatten details)
        if layer['class_name']=='Dense':
            layer['n_in']=weights.shape[0]
            layer['n_out']=weights.shape[1]
            # if this layer is too big (more than MAXMULT multiplications); 
            # break it out into chunks!
            layer['n_subout']=[weights.shape[1]]
            if layer['n_in']*layer['n_out']>MAXMULT and yamlConfig["IOType"] != "io_serial":
                n_subout = int(MAXMULT/layer['n_in'])
                n_totout = 0
                layer['n_subout'] = []
                layer['weights_n_subzeros'] = []
                layer['n_part'] = 0
                while n_totout < layer['n_out']:
                    if n_totout + n_subout <= layer['n_out']:
                        layer['n_subout'].append(n_subout)
                        n_totout += n_subout                    
                    else:
                        layer['n_subout'].append(layer['n_out']-n_totout)
                        n_totout += layer['n_out']-n_totout
                    layer['n_part'] += 1
                for i_part in range(0,layer['n_part']):
                    i_subout = 0
                    if i_part>0:
                        i_subout = sum(layer['n_subout'][0:i_part])
                    cur_n_zeros = print_array_to_cpp("w{}".format(layer_counter), weights, yamlConfig['OutputDir'], i_part, layer['n_part'], i_subout, layer['n_subout'][i_part])
                    print_array_to_cpp("b{}".format(layer_counter), biases, yamlConfig['OutputDir'], i_part, layer['n_part'], i_subout, layer['n_subout'][i_part])
                    layer['weights_n_subzeros'].append(cur_n_zeros)
            
            current_shape = [current_shape[0], layer['n_out']]
        elif layer['class_name']=='Conv1D':
            # weights.shape = (filter_width, n_channels, n_filters)
            layer['y_in']=current_shape[1]
            layer['y_filt']=weights.shape[0] # or keras_layer['config']['kernel_size']
            layer['n_chan']=weights.shape[1] 
            layer['n_filt']=weights.shape[2] # or keras_layer['config']['filters']
            layer['stride']=keras_layer['config']['strides'][0]
            layer['padding']=keras_layer['config']['padding']
            if layer['padding']=='same':
                in_width = current_shape[1]
                layer['y_out'] = int(math.ceil(float(in_width) / float(layer['stride'])))
                if (in_width % layer['stride'] == 0):
                    pad_along_width = max(layer['y_filt'] - layer['stride'], 0)
                else:
                    pad_along_width = max(layer['y_filt'] - (in_width % layer['stride']), 0)
                layer['pad_left']  = pad_along_width // 2
                layer['pad_right']  = pad_along_width - layer['pad_left']
            elif layer['padding']=='valid':
                in_width = current_shape[1]
                layer['y_out'] = int(math.ceil(float(in_width - layer['y_filt'] + 1) / float(layer['stride'])))
                layer['pad_left'] = 0
                layer['pad_right'] = 0
            current_shape=[current_shape[0], layer['y_out'], layer['n_filt']]
        elif layer['class_name']=='Conv2D':
            layer['in_height']=current_shape[1]
            layer['in_width']=current_shape[2]
            layer['filt_height']=weights.shape[0]
            layer['filt_width']=weights.shape[1]
            layer['n_chan']=weights.shape[2]
            layer['n_filt']=weights.shape[3]
            layer['stride_height']=keras_layer['config']['strides'][0]
            layer['stride_width']=keras_layer['config']['strides'][1]
            layer['padding']=keras_layer['config']['padding']
            if layer['padding']=='same':
                #Height
                in_height = current_shape[1]
                layer['out_height'] = int(math.ceil(float(in_height) / float(layer['stride_height'])))
                if (in_height % layer['stride_height'] == 0):
                    pad_along_height = max(layer['filt_height'] - layer['stride_height'], 0)
                else:
                    pad_along_height = max(layer['filt_height'] - (in_height % layer['stride_height']), 0)
                layer['pad_top']  = pad_along_height // 2
                layer['pad_bottom']  = pad_along_height - layer['pad_top']
                #Width
                in_width = current_shape[2]
                layer['out_width'] = int(math.ceil(float(in_width) / float(layer['stride_width'])))
                if (in_width % layer['stride_width'] == 0):
                    pad_along_width = max(layer['filt_width'] - layer['stride_width'], 0)
                else:
                    pad_along_width = max(layer['filt_width'] - (in_width % layer['stride_width']), 0)
                layer['pad_left']  = pad_along_width // 2
                layer['pad_right']  = pad_along_width - layer['pad_left']
            elif layer['padding']=='valid':
                in_height = current_shape[1]
                in_width = current_shape[2]
                layer['out_width'] = int(math.ceil(float(in_width - layer['filt_width'] + 1) / float(layer['stride_width'])))
                layer['out_height'] = int(math.ceil(float(in_height - layer['filt_height'] + 1) / float(layer['stride_height'])))
                layer['pad_top'] = 0
                layer['pad_bottom'] = 0
                layer['pad_left'] = 0
                layer['pad_right'] = 0
            current_shape=[current_shape[0], layer['out_height'], layer['out_width'], layer['n_filt']]
        elif layer['class_name']=='BatchNormalization':
            if is_dense:
                layer['n_in']=mean.shape[0]
                layer['n_out']=mean.shape[0]
                layer['n_filt'] = -1
                current_shape = [current_shape[0], layer['n_out']]
            elif is_conv2d:
                layer['n_in']=current_shape[1]*current_shape[2]*current_shape[3] 
                layer['n_out']=layer['n_in']
                layer['in_height']=current_shape[1]
                layer['in_width']=current_shape[2]
                layer['n_filt']=current_shape[3]
                current_shape=[current_shape[0], layer['in_height'], layer['in_width'], layer['n_filt']]
        elif 'Pooling' in layer['class_name']:
            info = layer['class_name'].split('Pooling')
            d = int(info[1].split('D')[0])
            op = info[0]
            if d == 1:
                layer['pool_size']=keras_layer['config']['pool_size']
                layer['stride']=keras_layer['config']['stride']
            elif d == 2:
                layer['in_height']=current_shape[1]
                layer['in_width']=current_shape[2]
                layer['n_filt']=layer_list[-1]['n_filt']
                layer['stride_height']=keras_layer['config']['strides'][0]
                layer['stride_width']=keras_layer['config']['strides'][1]
                layer['pool_height']=keras_layer['config']['pool_size'][0]
                layer['pool_width']=keras_layer['config']['pool_size'][1]
                layer['padding']=keras_layer['config']['padding']
            if layer['padding']=='same':
                #Height
                in_height = current_shape[1]
                layer['out_height'] = int(math.ceil(float(in_height) / float(layer['stride_height'])))
                if (in_height % layer['stride_height'] == 0):
                    pad_along_height = max(layer['pool_height'] - layer['stride_height'], 0)
                else:
                    pad_along_height = max(layer['pool_height'] - (in_height % layer['stride_height']), 0)
                layer['pad_top']  = pad_along_height // 2
                layer['pad_bottom']  = pad_along_height - layer['pad_top']
                #Width
                in_width = current_shape[2]
                layer['out_width'] = int(math.ceil(float(in_width) / float(layer['stride_width'])))
                if (in_width % layer['stride_width'] == 0):
                    pad_along_width = max(layer['pool_width'] - layer['stride_width'], 0)
                else:
                    pad_along_width = max(layer['pool_width'] - (in_width % layer['stride_width']), 0)
                layer['pad_left']  = pad_along_width // 2
                layer['pad_right']  = pad_along_width - layer['pad_left']
                layer['n_out'] = layer['out_height'] * layer['out_width'] * layer['n_filt']
            elif layer['padding']=='valid':
                in_height = current_shape[1]
                in_width = current_shape[2]
                layer['out_width'] = int(math.ceil(float(in_width - layer['pool_width'] + 1) / float(layer['stride_width'])))
                layer['out_height'] = int(math.ceil(float(in_height - layer['pool_height'] + 1) / float(layer['stride_height'])))
                layer['pad_top'] = 0
                layer['pad_bottom'] = 0
                layer['pad_left'] = 0
                layer['pad_right'] = 0
                layer['n_out'] = layer['out_height'] * layer['out_height'] * layer['n_filt'] 
            current_shape=[current_shape[0], layer['out_height'], layer['out_width'], layer['n_filt']]

        elif layer['class_name']=='Activation':
            if layer_list[-1]['class_name'] != 'BatchNormalization':
                layer_list[-1]['activation'] = layer['activation']
                skip_layer = True
                layer_counter = layer_counter - 1
        elif layer['class_name']=='LeakyReLU':
            if layer_list[-1]['class_name'] != 'BatchNormalization':
                layer_list[-1]['activation'] = layer['class_name']
                layer_list[-1]['activ_param'] = keras_layer["config"].get('alpha', 0.3)
                skip_layer = True
                layer_counter = layer_counter - 1
            else:
                layer['activation'] = layer['class_name']
                layer['activ_param'] = keras_layer["config"].get('alpha', 0.3)
        elif layer['class_name']=='ThresholdedReLU':
            if layer_list[-1]['class_name'] != 'BatchNormalization':
                layer_list[-1]['activation'] = layer['class_name']
                layer_list[-1]['activ_param'] = keras_layer["config"].get('theta', 1.)
                skip_layer = True
                layer_counter = layer_counter - 1
            else:
                layer['activation'] = layer['class_name']
                layer['activ_param'] = keras_layer["config"].get('theta', 1.)
        elif layer['class_name']=='ELU':
            if layer_list[-1]['class_name'] != 'BatchNormalization':
                layer_list[-1]['activation'] = layer['class_name']
                layer_list[-1]['activ_param'] = keras_layer["config"].get('alpha', 1.)
                skip_layer = True
                layer_counter = layer_counter - 1
            else:
                layer['activation'] = layer['class_name']
                layer['activ_param'] = keras_layer["config"].get('alpha', 1.)
        elif layer['class_name']=='PReLU':
            if layer_list[-1]['class_name'] != 'BatchNormalization':
                layer_list[-1]['activation'] = layer['class_name']
                skip_layer = True
                layer_counter = layer_counter - 1
            else:
                layer['activation'] = layer['class_name']
            
            #Translate learned alpha array from h5 file
            weights = h5File['/{}/{}/alpha:0'.format(layer['name'],layer['name'])][()]
            print_array_to_cpp("a{}".format(layer_counter), weights, yamlConfig['OutputDir'])

        if not skip_layer:
            print('Layer name: {}, layer type: {}, current shape: {}, number of zeros: {}'.format(layer['name'], layer['class_name'], current_shape, cur_n_zeros))
            if layer['n_part'] > 1: 
                print(' -> layer will be divided into {} sublayer calls; output neurons: {} '.format(layer['n_part'], layer['n_subout']))
            layer_list.append( layer )


    #################
    ## Generate HLS
    #################

    #Weights and biases are already dumped to output directory
    #Now generate HLS from list of layer dictionaries
    hls_writer(layer_list, yamlConfig)


if __name__ == "__main__":
    main()
