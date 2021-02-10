from hls4ml.model.optimizer import OptimizerPass


class FuseBatchNormalization(OptimizerPass):
    def match(self, node):
        is_match = node.__class__.__name__ == 'BatchNormalization' and \
            node.get_input_node().__class__.__name__ in ['Dense', 'Conv1D', 'Conv2D'] and \
            node.get_input_node().get_attr('weight_quantizer') is None and \
            node.get_input_node().get_attr('bias_quantizer') is None
        return is_match

    def transform(self, model, node):
        # Fuse weight and bias of Dense/Conv1D/Conv2D layer with BN values
        parent_node = node.get_input_node()

        parent_weight = parent_node.weights['weight']
        parent_bias = parent_node.weights['bias']

        bn_scale = node.weights['scale']
        bn_bias = node.weights['bias']

        if parent_node.get_attr('strategy') != 'resource':
            fused_weight = bn_scale.data * parent_weight.data
        else:
            fused_weight = (bn_scale.data * parent_weight.data.T).T
        fused_bias = bn_scale.data * parent_bias.data + bn_bias.data

        model.remove_node(node, rewire=True)
        parent_weight.data = fused_weight
        parent_bias.data = fused_bias
        if not parent_node.get_attr('use_bias', True):
            parent_bias.update_precision(bn_bias.type.precision)

        return True
