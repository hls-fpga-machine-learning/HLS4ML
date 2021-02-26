from hls4ml.model.optimizer import OptimizerPass


class EliminateLinearActivation(OptimizerPass):
    def match(self, node):
        cast = False
        if node.__class__.__name__ == 'Activation':
            cast = node.get_input_variable().type.precision != node.get_output_variable().type.precision
        return node.__class__.__name__ == 'Activation' and node.get_attr('activation') == 'linear' and not cast

    def transform(self, model, node):
        model.remove_node(node)
        return True
