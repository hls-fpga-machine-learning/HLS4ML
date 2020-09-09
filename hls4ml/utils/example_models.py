from urllib.request import urlretrieve
from .config import create_config
import pprint
import json

def fetch_example_model(model_name):
    """
    Download an example model from github repo to working directory, and return the corresponding configuration:

    https://github.com/hls-fpga-machine-learning/example-models

    Args:
        - model_name: string, name of the example model in the repo. Example: 'KERAS_3layer.json'
    
    """

    #Initilize the download link and model type
    download_link = 'https://raw.githubusercontent.com/hls-fpga-machine-learning/example-models/master/'
    model_type = None
    model_config = None

    #Check for model's type to update link
    if '.json' in model_name:
        model_type = 'keras'
        model_config = 'KerasJson'
    elif '.pt' in model_name:
        model_type = 'pytorch'
        model_config = 'PytorchModel'
    elif '.onnx' in model_name:
        model_type = 'onnx'
        model_config ='OnnxModel'
    elif '.pb' in model_name:
        model_type = 'tensorflow'
        model_config = 'TensorFlowModel'
    else:
        raise TypeError('Model type is not supported in hls4ml.')
    

    download_link_model = download_link + model_type + '/' + model_name

    #Initiate the configuration file
    config = create_config()
        
    #Download the example model
    urlretrieve(download_link_model, model_name)

    #If the model is a keras model then have to download its weight file as well
    if model_type == 'keras':
        model_weight_name = model_name[:-5] + "_weights.h5"

        download_link_weight = download_link + model_type + '/' + model_weight_name
        urlretrieve(download_link_weight, model_weight_name)

        config['KerasH5'] =  model_weight_name #Set configuration for the weight file

    #Additional configuration parameters
    config[model_config] = model_name
    config['HLSConfig']['Model'] = {}
    config['HLSConfig']['Model']['Precision'] = 'ap_fixed<16,6>'
    config['HLSConfig']['Model']['ReuseFactor'] = '1'
    
    return config

def fetch_example_list():
    
    link_to_list = 'https://raw.githubusercontent.com/hls-fpga-machine-learning/example-models/master/available_models.json'
    
    temp_file, _ = urlretrieve(link_to_list)
    
    # Read data from file:
    data = json.load(open(temp_file))
    
    # Print in fancy format
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(data)