# Heavily inspired by Keras's plot_model
"""Utilities related to model visualization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

try:
    import pydot
except ImportError:
    pydot = None


def check_pydot():
    """Returns True if PyDot and Graphviz are available."""
    if pydot is None:
        return False
    try:
        # Attempt to create an image of a blank graph
        # to check the pydot/graphviz installation.
        pydot.Dot.create(pydot.Dot())
        return True
    except OSError:
        return False


def add_edge(dot, src, dst):
    if not dot.get_edge(src, dst):
        dot.add_edge(pydot.Edge(src, dst))


def model_to_dot(model,
                 show_shapes=False,
                 show_layer_names=True,
                 show_precision=False,
                 rankdir='TB',
                 dpi=96,
                 subgraph=False):
    """Convert a HLS model to dot format.

    Arguments:
        model: A HLS model instance.
        show_shapes: whether to display shape information.
        show_layer_names: whether to display layer names.
        show_precision: whether to display precision of layer's variables.
        rankdir: `rankdir` argument passed to PyDot,
            a string specifying the format of the plot:
            'TB' creates a vertical plot;
            'LR' creates a horizontal plot.
        dpi: Dots per inch.
        subgraph: whether to return a `pydot.Cluster` instance.

    Returns:
        A `pydot.Dot` instance representing the HLS model or
        a `pydot.Cluster` instance representing nested model if
        `subgraph=True`.

    Raises:
        ImportError: if graphviz or pydot are not available.
    """

    if not check_pydot():
        if 'IPython.core.magics.namespace' in sys.modules:
            # We don't raise an exception here in order to avoid crashing notebook
            # tests where graphviz is not available.
            print('Failed to import pydot. You must install pydot'
                  ' and graphviz for `pydotprint` to work.')
            return
        else:
            raise ImportError('Failed to import pydot. You must install pydot'
                              ' and graphviz for `pydotprint` to work.')

    if subgraph:
        dot = pydot.Cluster(style='dashed', graph_name=model.name)
        dot.set('label', model.name)
        dot.set('labeljust', 'l')
    else:
        dot = pydot.Dot()
        dot.set('rankdir', rankdir)
        dot.set('concentrate', True)
        dot.set('dpi', dpi)
        dot.set_node_defaults(shape='record')

    layers = model.get_layers()

    # Create graph nodes.
    for i, layer in enumerate(layers):
        #layer_id = str(id(layer))
        layer_id = str(layer.index)

        # Append a wrapped layer's label to node's label, if it exists.
        layer_name = layer.name
        class_name = layer.__class__.__name__

        # Create node's label.
        if show_layer_names:
            #label = '{}: {}'.format(class_name, layer_name)
            #label = '{}\\l{}\\l'.format(class_name, layer_name)
            label = '<b>{}</b><br align="left" />{}'.format(class_name, layer_name)
        else:
            label = class_name

        # Rebuild the label as a table including input/output shapes.
        if show_shapes:
            def format_shape(shape):
                return str(tuple(shape)).replace(str(None), '?')

            input_labels = '?'
            try:
                output_labels = format_shape(layer.get_output_variable().shape)
            except AttributeError:
                output_labels = '?'
            if class_name != 'Input':
                if len(layer.inputs) > 1:
                    input_shapes = []
                    for i in layer.inputs:
                        input_layer = layer.get_input_variable(i)
                        if input_layer is not None:
                            input_shapes.append(input_layer.shape)
                        else:
                            input_shapes.append('?')
                    formatted_shapes = [format_shape(ishape) for ishape in input_shapes]
                    input_labels = ', '.join(formatted_shapes)
                else:
                    input_layer = layer.get_input_variable()
                    if input_layer is not None:
                        input_labels = format_shape(input_layer.shape)
            label = '%s\n|{input: %s|output: %s}' % (label,
                                                     input_labels,
                                                     output_labels)

        # Rebuild the label as a table including tensor precision.
        if show_precision:
            def format_precision(precision):
                return str(precision).replace('<', '&lt;').replace('>', '&gt;')

            precision_labels = []
            tensors = {}
            tensors.update(layer.weights)
            if len(layer.variables) == 1:
                # A bit cleaner output
                tensors['output'] = layer.get_output_variable()
            else:
                tensors.update(layer.variables)
            for tensor_name, var in tensors.items():
                if show_shapes:
                    # tensor_label = '{} {}: {}'.format(tensor_name,
                    tensor_label = '<tr><td align="left">{} {}:</td><td align="left">{}</td></tr>'.format(
                        tensor_name,
                        format_shape(var.shape),
                        format_precision(var.type.precision))
                else:
                    # tensor_label = '{}: {}'.format(tensor_name,
                    tensor_label = '<tr><td align="left">{}:</td><td align="left">{}</td></tr>'.format(
                        tensor_name,
                        format_precision(var.type.precision))
                precision_labels.append(tensor_label)
            #precision_label = '<br align="left" />'.join(precision_labels)
            precision_label = ''.join(precision_labels)
            precision_label = '<table border="0" cellspacing="0">' + precision_label + '</table>'
            label = '%s|{%s}' % (label, precision_label)

        label = '<' + label + '>'
        node = pydot.Node(layer_id, label=label)
        dot.add_node(node)

    # Connect nodes with edges.
    for layer in layers:
        layer_id = str(layer.index)
        for i, input_name in enumerate(layer.inputs):
            input_layer = layer.get_input_node(input_name)
            if input_layer is not None:
                input_layer_id = str(input_layer.index)
                add_edge(dot, input_layer_id, layer_id)

    return dot


def plot_model(model,
               to_file='model.png',
               show_shapes=False,
               show_layer_names=True,
               show_precision=False,
               rankdir='TB',
               dpi=96):
    """Converts a HLS model to dot format and save to a file.

    Arguments:
        model: A HLS model instance
        to_file: File name of the plot image.
        show_shapes: whether to display shape information.
        show_layer_names: whether to display layer names.
        show_precision: whether to display precision of layer's variables.
        rankdir: `rankdir` argument passed to PyDot,
            a string specifying the format of the plot:
            'TB' creates a vertical plot;
            'LR' creates a horizontal plot.
        dpi: Dots per inch.

    Returns:
        A Jupyter notebook Image object if Jupyter is installed.
        This enables in-line display of the model plots in notebooks.
    """
    dot = model_to_dot(model,
                       show_shapes=show_shapes,
                       show_layer_names=show_layer_names,
                       show_precision=show_precision,
                       rankdir=rankdir,
                       dpi=dpi)
    if dot is None:
        return

    if to_file is not None:
        _, extension = os.path.splitext(to_file)
        if not extension:
            extension = 'png'
        else:
            extension = extension[1:]
        # Save image to disk.
        dot.write(to_file, format=extension)
    else:
        # Return the image as a Jupyter Image object, to be displayed in-line.
        # Note that we cannot easily detect whether the code is running in a
        # notebook, and thus we always return the Image if Jupyter is available.
        try:
            from IPython import display
            import tempfile

            temp = tempfile.NamedTemporaryFile(suffix='.png')
            dot.write(temp.name, format='png')
            return display.Image(filename=temp.name)
        except ImportError:
            pass
