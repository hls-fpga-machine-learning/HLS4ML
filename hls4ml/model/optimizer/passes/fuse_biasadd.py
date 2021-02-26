from hls4ml.model.optimizer import OptimizerPass

from hls4ml.model import hls_model


class FuseBiasAdd(OptimizerPass):
    ''' Fuses BiasAdd into Dense/Conv2D layer (common in TF models). '''

    def match(self, node):
        is_match = node.__class__.__name__ == 'BiasAdd' and \
            (node.get_input_node().__class__.__name__ == 'Dense' or
             node.get_input_node().__class__.__name__ == 'Conv2D')
        return is_match

    def transform(self, model, node):
        # Fuse BiasAdd into Dense layer
        dense_layer = node.get_input_node()
        dense_layer.get_weights('bias').data = node.get_weights('bias').data

        model.remove_node(node, rewire=True)

        return True
