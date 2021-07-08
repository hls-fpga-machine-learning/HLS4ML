from __future__ import print_function
from sys import path_importer_cache
import numpy as np
import math
from onnx import ModelProto, GraphProto, NodeProto, TensorProto
from onnx import  helper, numpy_helper, shape_inference

from hls4ml.model import HLSModel

MAXMULT = 4096

class ONNXDataReader:
    """
    ONNX data reader to be used for extracting relevant information during conversion.
    """
    def __init__(self, model):
        self.model = model
        self.input_map = {}
        self.index_map = {
            # Dense
            'kernel' : 1,
            'bias'   : 2,
            # BatchNormalization
            'gamma'  : 1,
            'beta'   : 2,
            'moving_mean'   : 3,
            'moving_variance' : 4,
        }

    def get_weights_data(self, layer_name, var_name):
        """Extract weights data from ONNX model.
        
        Parameters
        ----------
        layer_name : string
            layer's name in the ONNX model
        var_name : string
            variable to be extracted

        Returns
        -------
        data : numpy array
            extracted weights data 
        
        """
        #Get the node associated with the layer name
        node = next((node for node in self.model.graph.node if node.name == layer_name))
        
        inputs = self.input_map[layer_name]
        inp_idx = self.index_map[var_name]
        
        if inp_idx >= len(inputs['inputs']):
            # Input not found, likely a bias tensor is not available
            return None

        tensor = next((x for x in self.model.graph.initializer if x.name == inputs['inputs'][inp_idx]), None)
        
        if tensor is not None:
            data = numpy_helper.to_array(tensor)
            if inputs['transpose']:
                if inputs['perm'] is not None and len(data.shape) == len(inputs['perm']):
                    data = data.transpose(inputs['perm'])
                else:
                    data = data.transpose()
            
            #Check for transB in Gemm
            if node.op_type == 'Gemm':
                if not get_onnx_attribute(node, 'transB'):
                    data = data.transpose()

        return data
    
    def add_input(self, layer_name, inputs, transpose=True, perm=None):
        self.input_map[layer_name] = { 'inputs': inputs, 'transpose': transpose, 'perm': perm }
    
####----------------------Helpers---------------------######
def sanitize_layer_name(layer):
    new_name = layer['name']
    if new_name[0].isdigit():
        new_name = layer['class_name'].lower() + new_name
    
    layer['name'] = new_name

def replace_char_inconsitency(name):
    """
    Replace some inconsistent characters that cause issues when writing into HLS.
    """
    return name.replace('.','_')

def get_onnx_attribute(operation, name, default=None):
    attr = next((x for x in operation.attribute if x.name == name), None)
    if attr is None:
        value = default
    else:
        value = helper.get_attribute_value(attr)
        if isinstance(value, bytes):
            value = value.decode()
    return value

def get_input_shape(model, operation, input_idx=0):
    value_info_idx = next((i for i, x in enumerate(model.graph.value_info) if x.name == operation.input[input_idx]), 0)
    return [d.dim_value for d in model.graph.value_info[value_info_idx].type.tensor_type.shape.dim]

def compute_pads_1d(operation, layer):
    auto_pad = get_onnx_attribute(operation, 'auto_pad', 'NOTSET')
    if auto_pad != 'NOTSET':
        if (layer['y_in'] % layer['stride'] == 0):
            pad_along_width = max(layer['y_filt'] - layer['stride'], 0)
        else:
            pad_along_width = max(layer['y_filt'] - (layer['y_in'] % layer['stride']), 0)

        pads = [pad_along_width // 2, pad_along_width - (pad_along_width // 2)]

        if auto_pad == 'SAME_UPPER':
            pads = sorted(pads)
        elif auto_pad == 'SAME_LOWER':
            pads = sorted(pads, reverse=True)
        else: # 'VALID' padding
            pads = [0, 0]
    else:
        pads = get_onnx_attribute(operation, 'pads', [0, 0])
    
    return pads

def compute_pads_2d(operation, layer):
    auto_pad = get_onnx_attribute(operation, 'auto_pad', 'NOTSET')
    if auto_pad != 'NOTSET':
        #Height
        if (layer['in_height'] % layer['stride_height'] == 0):
            pad_along_height = max(layer['filt_height'] - layer['stride_height'], 0)
        else:
            pad_along_height = max(layer['filt_height'] - (layer['in_height'] % layer['stride_height']), 0)
        pad_height = [pad_along_height // 2, pad_along_height - pad_along_height // 2]

        #Width
        if (layer['in_width'] % layer['stride_width'] == 0):
            pad_along_width = max(layer['filt_width'] - layer['stride_width'], 0)
        else:
            pad_along_width = max(layer['filt_width'] - (layer['in_width'] % layer['stride_width']), 0)
        pad_width = [pad_along_width // 2, pad_along_width - pad_along_width // 2]

        if auto_pad == 'SAME_UPPER':
            pads = [min(pad_height), min(pad_width), max(pad_height), max(pad_width)]
        elif auto_pad == 'SAME_LOWER':
            pads = [max(pad_height), max(pad_width), min(pad_height), min(pad_width)]
        else: # 'VALID' padding
            pads = [0, 0, 0, 0]
    else:
        pads = get_onnx_attribute(operation, 'pads', [0, 0, 0, 0])
    
    return pads

####----------------------Layer handling---------------------######
layer_handlers = {}

def register_onnx_layer_handler(layer_name, handler_func):
    if layer_name in layer_handlers:
        raise Exception('Layer {} already registered'.format(layer_name))
    else:
        layer_handlers[layer_name] = handler_func

def get_supported_onnx_layers():
    return list(layer_handlers.keys())

def onnx_handler(*args):
    def decorator(function):
        function.handles = [arg for arg in args]
        return function
    return decorator

#--->> A set of functions to address the naming convetion in ONNx's graph
def get_onnx_input_name(node, graph):
    """
    In ONNX, when calling node.input, it returns the node input's index in the graph instead of the input's name.
    However, the input's name is used for indexing in HLSModel's graph. This function return the input node's name instead.
    """
    input_node_name = [in_node.name for in_node in graph.node if (in_node.output[0] == node.input[0])]

    if input_node_name:
        return input_node_name
    else: #If there is no input name it's actually the first layer
        return [replace_char_inconsitency(node.input[0])]

def get_out_layer_name(graph):
    """
    Get the output layer's name for the model.
    graph.output only returns the output's node index
    """
    output_index_list = [x.name for x in graph.output]
    return [node.name for node in graph.node if node.output[0] in output_index_list]


def onnx_to_hls(config):
    """ Convert onnx model to hls model from configuration.
    
    Parameters
    ----------
    config: dict
        onnx configuration from yaml file or passed through API.
        
    Returns
    -------
    hls_model : hls4ml model object
        
    """

    #This is a list of dictionaries to hold all the layer info we need to generate HLS
    layer_list = []

    #Extract model architecture
    print('Interpreting Model ...')
    
    model =  onnx.load(config['OnnxModel']) if isinstance(config['OnnxModel'], str) else config['OnnxModel']
          
    model = shape_inference.infer_shapes(model)
    graph = model.graph
    
    reader = ONNXDataReader(model)
    
    #Obtain list of input/ouput layers
    all_inputs = [x.name for x in model.graph.input]
    all_initializers = [x.name for x in model.graph.initializer]
    input_layers = [x for x in all_inputs if x not in all_initializers]
    output_layers = get_out_layer_name(graph)
    
    print("Output layers: ", output_layers)
    
    for i, inp in enumerate(input_layers):
        input_layer = {}
        input_layer['name'] = replace_char_inconsitency(inp)
        input_layer['class_name'] = 'InputLayer'
        inp_shape = next((x.type.tensor_type.shape.dim for x in model.graph.input if x.name == inp), None)
        input_layer['input_shape'] = [x.dim_value for x in inp_shape]
        
        if len(input_layer['input_shape']) > 1:
            input_layer['input_shape'][0] = None #Firt dim is batch

        #Clean the layer name for specific models
        sanitize_layer_name(input_layer)
        input_layers[i] = input_layer['name']

        layer_list.append(input_layer)

    # Defined supported layers and check for unsupported layer type
    skip_layers = ['Dropout', 'Identity', 'Flatten']
    
    #Map inputs of skipped layers
    inputs_map = {}
    
    supported_layers = get_supported_onnx_layers() + skip_layers
    
    # Get input shape
    current_shape = [input_layer['input_shape']]
    print('Input shape:', current_shape[0])
    
    #Loop through layers
    layer_counter = 0
    
    #Output shape tracking
    output_shapes = {}
    output_shape = None

    print('Topology:')
    for node in graph.node:
        
        if node.op_type not in supported_layers:
            raise Exception('ERROR: Unsupported operation type: {}'.format(node.op_type))
        
        #If not the first layer then input shape is taken from last layer's output
        if layer_counter != 0:
            current_shape = [output_shape]
            
        if node.op_type in skip_layers:
            if node.op_type == 'Flatten':
                output_shape = [current_shape[0][0], np.prod(current_shape[0][1:])]
            
            else:
                #Currently supported skipped layers have only one input and output
                #Skipped layers can follow each other (e.g., Dropout -> Flatten)
                
                #Mapping inputs
                input_name = inputs_map.get(node.input[0], node.input[0])
                output_name = node.output[0]
                inputs_map[output_name] = input_name
                
                output_shape = current_shape[0]
            continue 
        
        if node.op_type in supported_layers:
            layer_counter = layer_counter + 1
        
        #Process the layer
        layer, output_shape = layer_handlers[node.op_type](reader, node, inputs_map, current_shape, graph, config)
        
        sanitize_layer_name(layer)
        print('Layer name: {}, layer type: {}, current shape: {}'.format(layer['name'], layer['class_name'], current_shape))
        layer_list.append(layer)


    #################
    ## Generate HLS
    #################

    print('Creating HLS model')
    hls_model = HLSModel(config, reader, layer_list, input_layers, output_layers)
    return hls_model
