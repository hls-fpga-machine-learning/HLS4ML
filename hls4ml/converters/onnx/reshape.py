from hls4ml.converters.onnx_to_hls import onnx_handler, get_onnx_input_name

@onnx_handler('Transpose')
def parse_transpose_layer(reader, node, inputs_map, input_shapes, graph, config):
    
    layer = {}
    layer['name'] = node.name
    layer['class_name'] = 'Transpose'

    layer['inputs'] = get_onnx_input_name(node, graph)
    
    perm = [list(i.ints) for i in node.attribute][0] #This will get something like [[a,b,c]][0] = [a,b,c]    
    layer['perm'] = perm
    
    output_shape = [input_shapes[0][i] for i in perm]
    
    return layer, output_shape

