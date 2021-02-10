from urllib.request import urlretrieve
from .config import create_vivado_config
import pprint
import json
import yaml


def _load_data_config_avai(model_name):
    """
    Check data and configuration availability for each model from this file:

    https://github.com/hls-fpga-machine-learning/example-models/blob/master/available_data_config.json
    """

    link_to_list = 'https://raw.githubusercontent.com/hls-fpga-machine-learning/example-models/master/available_data_config.json'

    temp_file, _ = urlretrieve(link_to_list)

    # Read data from file:
    data = json.load(open(temp_file))

    return data[model_name]


def _data_is_available(model_name):

    data = _load_data_config_avai(model_name)

    return data['example_data']


def _config_is_available(model_name):

    data = _load_data_config_avai(model_name)

    return data['example_config']


def _create_default_config(model_name, model_config):

    # Initiate the configuration file
    config = create_vivado_config()

    # Additional configuration parameters
    config[model_config] = model_name
    config['HLSConfig']['Model'] = {}
    config['HLSConfig']['Model']['Precision'] = 'ap_fixed<16,6>'
    config['HLSConfig']['Model']['ReuseFactor'] = '1'

    return config


def _filter_name(model_name):
    """
    Need to get "garnet_1layer" from "garnet_1layer.json" for loading of data and configuration files
    """
    filtered_name = None

    if model_name.endswith('.json') or model_name.endswith('.onnx'):
        filtered_name = model_name[:-5]
    elif model_name.endswith('.pt') or model_name.endswith('.pb'):
        filtered_name = model_name[:-3]

    return filtered_name


def _load_example_data(model_name):

    print("Downloading input & output example files ...")

    filtered_name = _filter_name(model_name)

    input_file_name = filtered_name + "_input.dat"
    output_file_name = filtered_name + "_output.dat"

    link_to_input = 'https://raw.githubusercontent.com/hls-fpga-machine-learning/example-models/master/data/' + input_file_name
    link_to_output = 'https://raw.githubusercontent.com/hls-fpga-machine-learning/example-models/master/data/' + output_file_name

    urlretrieve(link_to_input, input_file_name)
    urlretrieve(link_to_output, output_file_name)


def _load_example_config(model_name):

    print("Downloading configuration files ...")

    filtered_name = _filter_name(model_name)

    config_name = filtered_name + "_config.yml"

    link_to_config = 'https://raw.githubusercontent.com/hls-fpga-machine-learning/example-models/master/config-files/' + config_name

    # Load the configuration as dictionary from file
    urlretrieve(link_to_config, config_name)

    # Load the configuration from local yml file
    with open(config_name, 'r') as ymlfile:
        config = yaml.load(ymlfile)

    return config


def fetch_example_model(model_name):
    """
    Download an example model (and example data & configuration if available) from github repo to working directory, and return the corresponding configuration:

    https://github.com/hls-fpga-machine-learning/example-models

    Use fetch_example_list() to see all the available models.

    Args:
        - model_name: string, name of the example model in the repo. Example: fetch_example_model('KERAS_3layer.json')

    Return:
        - Dictionary that stores the configuration to the model
    """

    # Initilize the download link and model type
    download_link = 'https://raw.githubusercontent.com/hls-fpga-machine-learning/example-models/master/'
    model_type = None
    model_config = None

    # Check for model's type to update link
    if '.json' in model_name:
        model_type = 'keras'
        model_config = 'KerasJson'
    elif '.pt' in model_name:
        model_type = 'pytorch'
        model_config = 'PytorchModel'
    elif '.onnx' in model_name:
        model_type = 'onnx'
        model_config = 'OnnxModel'
    elif '.pb' in model_name:
        model_type = 'tensorflow'
        model_config = 'TensorFlowModel'
    else:
        raise TypeError('Model type is not supported in hls4ml.')

    download_link_model = download_link + model_type + '/' + model_name

    # Download the example model
    print("Downloading example model files ...")
    urlretrieve(download_link_model, model_name)

    # Check if the example data and configuration for the model are available
    if _data_is_available(model_name):
        _load_example_data(model_name)

    if _config_is_available(model_name):
        config = _load_example_config(model_name)
    else:
        config = _create_default_config(model_name, model_config)

    # If the model is a keras model then have to download its weight file as well
    if model_type == 'keras':
        model_weight_name = model_name[:-5] + "_weights.h5"

        download_link_weight = download_link + model_type + '/' + model_weight_name
        urlretrieve(download_link_weight, model_weight_name)

        config['KerasH5'] = model_weight_name  # Set configuration for the weight file

    return config


def fetch_example_list():

    link_to_list = 'https://raw.githubusercontent.com/hls-fpga-machine-learning/example-models/master/available_models.json'

    temp_file, _ = urlretrieve(link_to_list)

    # Read data from file:
    data = json.load(open(temp_file))

    # Print in fancy format
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(data)
