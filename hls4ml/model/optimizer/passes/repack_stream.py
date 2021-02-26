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


repack_function_template = 'nnet::repack_stream<{input_t}, {output_t}, {size}>({input}, {output});'
repack_include_list = ['nnet_utils/nnet_stream.h']

# Register the layer types to the layer map
register_layer('Repack', Repack)

# Register the templates for config and function
templates.get_backend('Vivado').register_templates('Repack', repack_function_template, None, repack_include_list)


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
