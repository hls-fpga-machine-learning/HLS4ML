import numpy as np

from hls4ml.model.optimizer import OptimizerPass

from hls4ml.model.hls_model import Layer, register_layer
from hls4ml.templates import templates

class Repack(Layer):
    ''' Inserted between layers with different packing factors.'''

    def initialize(self):
        shape = self.attributes['target_shape']
        if shape[0] is None:
            shape = shape[1:]
        dims = ['N_SIZE_{}_{}'.format(i, self.index) for i in range(1, len(shape) + 1)]

        self.add_output_variable(shape, dims)

    def function_cpp(self):
        params = self._default_function_params()
        params['size'] = np.prod(self.get_output_variable().shape)
        return [self._function_template.format(**params)]

    def config_cpp(self):
        return None

class Broadcast(Layer):
    ''' Inserted between layers for broadcasting.'''

    def initialize(self):
        shape = self.attributes['target_shape']
        if shape[0] is None:
            shape = shape[1:]
        dims = ['N_SIZE_{}_{}'.format(i, self.index) for i in range(1, len(shape) + 1)]
        self.add_output_variable(shape, dims)

    def function_cpp(self):
        params = self._default_function_params()
        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()
        params['in_height'] = self.get_input_variable().shape[0]
        params['in_width'] = self.get_input_variable().shape[1]
        params['n_chan'] = self.get_input_variable().shape[2]
        params['n_dupl'] = int(np.prod(self.get_output_variable().shape)/np.prod(self.get_input_variable().shape))
        return self._config_template.format(**params)

repack_function_template = 'nnet::repack_stream<{input_t}, {output_t}, {size}>({input}, {output});'
repack_include_list = ['nnet_utils/nnet_stream.h']

broadcast_function_template = 'nnet::broadcast_stream<{input_t}, {output_t}, {config}>({input}, {output});'
broadcast_config_template = """struct config{index} : nnet::broadcast_config {{
static const unsigned in_width = {in_width};
static const unsigned in_height = {in_height};
static const unsigned n_chan = {n_chan};
static const unsigned n_dupl = {n_dupl};
}};\n"""
broadcast_include_list = ['nnet_utils/nnet_stream.h']

# Register the layer types to the layer map
register_layer('Repack', Repack)
register_layer('Broadcast', Broadcast)

# Register the templates for config and function
templates.get_backend('Vivado').register_templates('Repack', repack_function_template, None, repack_include_list)
templates.get_backend('Vivado').register_templates('Broadcast', broadcast_function_template, broadcast_config_template, broadcast_include_list)


class ReshapeStream(OptimizerPass):
    ''' Repacks stream for Reshape layer '''
    def match(self, node):
        return node.__class__.__name__ == 'Reshape'

    def transform(self, model, node):
        if model.config.backend.name != 'Vivado' or \
            model.config.get_config_value('IOType') != 'io_stream':
            return False

        attrs = {
            'target_shape': node.get_attr('target_shape')
        }

        # Insert new Repack node instead of Reshape
        repack_layer = model.make_node('Repack', 'repack_' + node.name, attrs, node.inputs.copy())
        model.replace_node(node, repack_layer)

        return True

class BroadcastStream(OptimizerPass):
    def match(self, node):
        if node.__class__.__name__ == 'Merge':
            try: 
                inp1 = node.get_input_variable(node.inputs[0])
                inp2 = node.get_input_variable(node.inputs[1])
                return inp1.shape != inp2.shape
            except:
                return False
        else:
            return False
        
    def transform(self, model, node):
        if model.config.backend.name not in ['Vivado'] or \
            model.config.get_config_value('IOType') != 'io_stream':
            return False
            
        inp1 = node.get_input_variable(node.inputs[0])
        inp2 = node.get_input_variable(node.inputs[1])
        if np.prod(inp1.shape) > np.prod(inp2.shape):
            idx = 1
            attrs = {
                'target_shape': inp1.shape
            }
        else:
            idx = 0
            attrs = {
                'target_shape': inp2.shape
            }
        brdcst_layer = model.make_node('Broadcast', 'broadcast_' + node.inputs[idx], attrs, [node.inputs[idx]].copy())
        model.insert_node(brdcst_layer)
        node.inputs[idx] = 'broadcast_' + node.inputs[idx]

        return True
