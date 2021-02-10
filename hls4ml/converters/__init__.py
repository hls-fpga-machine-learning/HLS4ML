from __future__ import absolute_import
import os
import importlib

from hls4ml.utils.config import create_vivado_config

from hls4ml.converters.keras_to_hls import keras_to_hls, get_supported_keras_layers, register_keras_layer_handler

for module in os.listdir(os.path.dirname(__file__) + '/keras'):
    if module == '__init__.py' or module[-3:] != '.py':
        continue
    try:
        lib = importlib.import_module(__name__ + '.keras.' + module[:-3])
        for name, func in list(lib.__dict__.items()):
            # if 'func' is callable (i.e., function, class...)
            # and has 'handles' attribute
            # and is defined in this module (i.e., not imported)
            if callable(func) and hasattr(func, 'handles') and func.__module__ == lib.__name__:
                for layer in func.handles:
                    register_keras_layer_handler(layer, func)
    except ImportError:
        continue

try:
    from hls4ml.converters.pytorch_to_hls import pytorch_to_hls
    __pytorch_enabled__ = True
except ImportError:
    __pytorch_enabled__ = False

try:
    from hls4ml.converters.onnx_to_hls import onnx_to_hls
    __onnx_enabled__ = True
except ImportError:
    __onnx_enabled__ = False

try:
    from hls4ml.converters.tf_to_hls import tf_to_hls
    __tensorflow_enabled__ = True
except ImportError:
    __tensorflow_enabled__ = False


def convert_from_yaml_config(yamlConfig):
    model = None
    if 'OnnxModel' in yamlConfig:
        if __onnx_enabled__:
            model = onnx_to_hls(yamlConfig)
        else:
            raise Exception("ONNX not found. Please install ONNX.")
    elif 'PytorchModel' in yamlConfig:
        if __pytorch_enabled__:
            model = pytorch_to_hls(yamlConfig)
        else:
            raise Exception("PyTorch not found. Please install PyTorch.")
    elif 'TensorFlowModel' in yamlConfig:
        if __tensorflow_enabled__:
            model = tf_to_hls(yamlConfig)
        else:
            raise Exception("TensorFlow not found. Please install TensorFlow.")
    else:
        model = keras_to_hls(yamlConfig)

    return model


def convert_from_keras_model(model, output_dir='my-hls-test', project_name='myproject',
                             fpga_part='xcku115-flvb2104-2-i', clock_period=5, io_type='io_parallel', hls_config={}):

    config = create_vivado_config(
        output_dir=output_dir,
        project_name=project_name,
        fpga_part=fpga_part,
        clock_period=clock_period,
        io_type=io_type
    )
    config['KerasModel'] = model

    model_config = hls_config.get('Model', None)
    if model_config is not None:
        if not all(k in model_config for k in ('Precision', 'ReuseFactor')):
            raise Exception('Precision and ReuseFactor must be provided in the hls_config')
    else:
        model_config = {}
        model_config['Precision'] = 'ap_fixed<16,6>'
        model_config['ReuseFactor'] = '1'
    config['HLSConfig']['Model'] = model_config

    if 'LayerName' in hls_config:
        config['HLSConfig']['LayerName'] = hls_config['LayerName']

    if 'LayerType' in hls_config:
        config['HLSConfig']['LayerType'] = hls_config['LayerType']

    if 'Optimizers' in hls_config:
        config['HLSConfig']['Optimizers'] = hls_config['Optimizers']

    if 'SkipOptimizers' in hls_config:
        config['HLSConfig']['SkipOptimizers'] = hls_config['SkipOptimizers']

    return keras_to_hls(config)
