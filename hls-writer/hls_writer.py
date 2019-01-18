from __future__ import print_function
import tarfile
import yaml
from shutil import copyfile
import numpy as np
import os
import re

def hls_writer(layer_list, yamlConfig):

    filedir = os.path.dirname(os.path.abspath(__file__))

    ###################
    ## myproject.cpp
    ###################

    f = open(os.path.join(filedir,'../hls-template/firmware/myproject.cpp'),'r')
    fout = open('{}/firmware/{}.cpp'.format(yamlConfig['OutputDir'], yamlConfig['ProjectName']),'w')

    # Set some variables to make the routine after a bit smoother
    do_batchnorm = False
    is_dense = False
    is_conv2d = False
    for i in range(1,len(layer_list)+1):
     if layer_list[i-1]['class_name'] == 'BatchNormalization': do_batchnorm = True
    for i in range(1,len(layer_list)+1):
     if layer_list[i-1]['class_name']=='Conv2D':
      is_conv2d = True
      break
    if not is_conv2d:
     for i in range(1,len(layer_list)+1):
      if layer_list[i-1]['class_name']=='Dense':
       is_dense = True
       break
    
    activation_layers = ['Activation', 'LeakyReLU', 'ThresholdedReLU', 'ELU', 'PReLU']
    
    # lines to add to .cpp for sublayers
    sublayerlines = []
    # lines to add to .h for sublayers
    sublayerlines_h = []
    for line in f.readlines():
        #Add headers to weights and biases
        if 'myproject' in line:
            newline = line.replace('myproject',yamlConfig['ProjectName'])
        elif 'input_t data[N_INPUTS]' in line and layer_list[0]['class_name']=='Conv1D':
            newline = line.replace('input_t data[N_INPUTS]','input_t data[Y_INPUTS_1][N_CHAN_1]')
        elif 'input_t data[N_INPUTS]' in line and layer_list[0]['class_name']=='Conv2D':
            newline = line.replace('input_t data[N_INPUTS]','input_t data[IN_HEIGHT_1][IN_WIDTH_1][N_CHAN_1]')
        elif 'input_t data[N_INPUTS]' in line and layer_list[0]['class_name']=='BatchNormalization' and is_conv2d:
            newline = line.replace('input_t data[N_INPUTS]','input_t data[IN_HEIGHT_1][IN_WIDTH_1][N_FILT_1]')
        elif 'const_size_in   = N_INPUTS' in line and layer_list[0]['class_name']=='Conv1D':
            newline = line.replace('const_size_in   = N_INPUTS','const_size_in   = Y_INPUTS_1*N_CHAN_1')
        elif 'const_size_in   = N_INPUTS' in line and layer_list[0]['class_name']=='Conv2D':
            newline = line.replace('const_size_in   = N_INPUTS','const_size_in   = IN_HEIGHT_1*IN_WIDTH_1*N_CHAN_1')
        elif 'const_size_in   = N_INPUTS' in line and layer_list[0]['class_name']=='BatchNormalization' and is_conv2d:
            newline = line.replace('const_size_in   = N_INPUTS','const_size_in   = IN_HEIGHT_1*IN_WIDTH_1*N_FILT_1')
        elif '//hls-fpga-machine-learning insert weights' in line:
            newline = line
            for i in range(1,len(layer_list)+1):
                if layer_list[i-1]['class_name'] == 'BatchNormalization':
                    newline += '#include "weights/beta{}.h"\n'.format(i)
                    newline += '#include "weights/scale{}.h"\n'.format(i)
                    newline += '#include "weights/mean{}.h"\n'.format(i)
                elif 'Pooling' in layer_list[i-1]['class_name']:
                    pass # No weights for pooling
                else:
                    if layer_list[i-1]['n_part']>1:
                        for i_part in range(layer_list[i-1]['n_part']):
                            newline += '#include "weights/w{}_{}.h"\n'.format(i,i_part)
                            newline += '#include "weights/b{}_{}.h"\n'.format(i,i_part)
                    elif layer_list[i-1]['class_name'] not in activation_layers:
                        newline += '#include "weights/w{}.h"\n'.format(i)
                        newline += '#include "weights/b{}.h"\n'.format(i)
                        if layer_list[i-1].get('activation') == 'PReLU':
                            newline += '#include "weights/a{}.h"\n'.format(i)
                    elif layer_list[i-1]['class_name'] == 'PReLU':
                        newline += '#include "weights/a{}.h"\n'.format(i)

        #Add input/output type
        elif '//hls-fpga-machine-learning insert IO' in line:
            newline = line
            if yamlConfig["IOType"] == "io_parallel":
                newline += '    #pragma HLS ARRAY_RESHAPE variable=data complete dim=0 \n'
                newline += '    #pragma HLS ARRAY_RESHAPE variable=res complete dim=0 \n'
                newline += '    #pragma HLS INTERFACE ap_vld port=data,res \n'
                newline += '    #pragma HLS DATAFLOW \n'
            if yamlConfig["IOType"] == "io_serial":
                newline += '    #pragma HLS INTERFACE axis port=data,res \n'
                newline += '    #pragma HLS DATAFLOW \n'

        #Add layers
        elif '//hls-fpga-machine-learning insert layers' in line:
            newline = line + '\n'
            for i in range(1,len(layer_list)+1):
                
                #Input to compute_layer

                #First layer and dense
                if i==1 and (layer_list[i-1]['class_name']=='Dense' or (layer_list[i-1]['class_name']=='BatchNormalization' and is_dense)):
                    input_type = 'input_t'
                    input_object = 'data'
                    n_in = 'N_INPUTS'
                #Layer is Dense and previous layer was Conv1D
                elif layer_list[i-1]['class_name']=='Dense' and layer_list[i-2]['class_name']=='Conv1D':
                    input_type = 'layer{}_t'.format(i-1)
                    input_object = 'layer{}_out'.format(i-1)
                    n_in = 'Y_OUTPUTS_{}*N_FILT_{}'.format(i-1,i-1)
                #Layer is Dense and previous layer was Conv2D
                elif layer_list[i-1]['class_name']=='Dense' and layer_list[i-2]['class_name']=='Conv2D':
                    input_type = 'layer{}_t'.format(i-1)
                    input_object = 'layer{}_out'.format(i-1)
                    n_in = 'IN_HEIGHT_{}*IN_WIDTH_{}*N_FILT_{}'.format(i-1,i-1,i-1)
                #Layer is Dense, BatchNormalization or Activation
                elif layer_list[i-1]['class_name']=='Dense' or layer_list[i-1]['class_name'] in activation_layers:
                    input_type = 'layer{}_t'.format(i-1)
                    input_object = 'layer{}_out'.format(i-1)
                    n_in = 'N_LAYER_{}'.format(i-1)
                elif is_dense and layer_list[i-1]['class_name']=='BatchNormalization':
                    input_type = 'layer{}_t'.format(i-1)
                    input_object = 'layer{}_out'.format(i-1)
                    n_in = 'N_LAYER_{}'.format(i-1)
                    n_filt = 'N_FILT_{}'.format(i-1)
                elif (i==1 and layer_list[i-1]['class_name']=='BatchNormalization' and is_conv2d):
                    input_type = 'input_t'
                    input_object = 'data'
                    in_height = 'IN_HEIGHT_{}'.format(i)
                    in_width = 'IN_WIDTH_{}'.format(i)
                    n_chan = 'N_FILT_{}'.format(i)
                elif is_conv2d and layer_list[i-1]['class_name']=='BatchNormalization':
                    input_type = 'layer{}_t'.format(i-1)
                    input_object = 'layer{}_out'.format(i-1)
                    n_in = 'OUT_HEIGHT_{}*OUT_WIDTH_{}*N_FILT_{}'.format(i-1,i-1,i-1)
                    n_filt = 'N_FILT_{}'.format(i-1)
                #First layer and Conv1D
                elif (i==1 and layer_list[i-1]['class_name']=='Conv1D'):
                    input_type = 'input_t'
                    input_object = 'data'
                    y_in = 'Y_INPUTS_{}'.format(i)
                    n_chan = 'N_CHAN_{}'.format(i)
                #Layer is Conv1D
                elif layer_list[i-1]['class_name']=='Conv1D':
                    input_type = 'layer{}_t'.format(i-1)
                    input_object = 'layer{}_out'.format(i-1)
                    y_in = 'Y_INPUTS_{}'.format(i)
                    n_chan = 'N_CHAN_{}'.format(i)
                #First layer and Conv2D
                elif (i==1 and layer_list[i-1]['class_name']=='Conv2D'):
                    input_type = 'input_t'
                    input_object = 'data'
                    in_height = 'IN_HEIGHT_{}'.format(i)
                    in_width = 'IN_WIDTH_{}'.format(i)
                    n_chan = 'N_CHAN_{}'.format(i)
                #Layer is Conv2D
                elif layer_list[i-1]['class_name']=='Conv2D':
                    input_type = 'layer{}_t'.format(i-1)
                    input_object = 'layer{}_out'.format(i-1)
                    in_height = 'IN_HEIGHT_{}'.format(i)
                    in_width = 'IN_WIDTH_{}'.format(i)
                    n_chan = 'N_CHAN_{}'.format(i)
                #Pooling layer
                elif 'Pooling' in layer_list[i-1]['class_name']:
                    input_type = 'layer{}_t'.format(i-1)
                    input_object = 'layer{}_out'.format(i-1)
                    output_object = 'layer{}_out'.format(i)
                    in_height = 'IN_HEIGHT_{}'.format(i)
                    in_width = 'IN_WIDTH_{}'.format(i)
                    out_height = 'OUT_HEIGHT_{}'.format(i)
                    out_width = 'OUT_WIDTH_{}'.format(i)
                    n_filt = 'N_FILT_{}'.format(i)
                #Currently doesn't allow all combinations


                #Outputs of compute_layer and activation 
                if i==len(layer_list) and layer_list[i-1]['class_name']=='Dense':
                    output_type = 'result_t'
                    output_object = 'res'
                    n_out = 'N_OUTPUTS'
                    if layer_list[i-1]['class_name'] in activation_layers: input_type = 'result_t'
                elif i==len(layer_list) and layer_list[i-1]['class_name'] in activation_layers and is_dense:
                    output_type = 'result_t'
                    output_object = 'res'
                    n_out = 'N_OUTPUTS'
                    input_type = 'result_t'
                elif i==len(layer_list) and is_dense and layer_list[i-1]['class_name']=='BatchNormalization':
                    output_type = 'result_t'
                    output_object = 'res'
                    n_out = 'N_OUTPUTS'
                elif i==len(layer_list) and is_conv2d and layer_list[i-1]['class_name']=='BatchNormalization':
                    output_type = 'layer{}_t'.format(i)
                    output_object = 'layer{}_out'.format(i)
                    out_height = 'OUT_HEIGHT_{}'.format(i)
                    out_width = 'OUT_WIDTH_{}'.format(i)
                    n_filt = 'N_FILT_{}'.format(i)
                elif(i==len(layer_list)-1 and is_dense and layer_list[i-1]['class_name']=='BatchNormalization' and layer_list[i]['class_name'] in activation_layers):
                    output_type = 'result_t'
                    output_object = 'layer{}_out'.format(i)
                    n_out = 'N_OUTPUTS' 
                elif layer_list[i-1]['class_name']=='Dense' or (layer_list[i-1]['class_name']=='BatchNormalization' and is_dense) or (layer_list[i-1]['class_name'] in activation_layers and is_dense):
                    output_type = 'layer{}_t'.format(i)
                    output_object = 'layer{}_out'.format(i)
                    n_out = 'N_LAYER_{}'.format(i)
                elif layer_list[i-1]['class_name']=='Conv1D':
                    output_type = 'layer{}_t'.format(i)
                    output_object = 'layer{}_out'.format(i)
                    y_out = 'Y_OUTPUTS_{}'.format(i)
                    n_filt = 'N_FILT_{}'.format(i)
                elif layer_list[i-1]['class_name']=='Conv2D' or (is_conv2d and layer_list[i-1]['class_name']=='BatchNormalization'):
                    output_type = 'layer{}_t'.format(i)
                    output_object = 'layer{}_out'.format(i)
                    out_height = 'OUT_HEIGHT_{}'.format(i)
                    out_width = 'OUT_WIDTH_{}'.format(i)
                    n_filt = 'N_FILT_{}'.format(i)
                #Currently assumes end with dense

                if( i!=len(layer_list) ):
                    if layer_list[i-1]['class_name']=='Dense' or (layer_list[i-1]['class_name']=='BatchNormalization' and is_dense) or (layer_list[i-1]['class_name'] in activation_layers and is_dense):
                        newline += '    {} layer{}_out[{}];\n'.format(output_type,i,n_out)
                    elif layer_list[i-1]['class_name']=='Conv1D' or 'Pooling1D' in layer_list[i-1]['class_name']:
                        newline += '    {} layer{}_out[{}*{}];\n'.format(output_type,i,y_out,n_filt)
                    elif layer_list[i-1]['class_name']=='Conv2D' or 'Pooling2D' in layer_list[i-1]['class_name']:
                        newline += '    {} layer{}_out[{}*{}*{}];\n'.format(output_type,i,out_height,out_width,n_filt)
                    elif layer_list[i-1]['class_name']=='BatchNormalization' and is_conv2d:
                        if i!= 1: newline += '    {} layer{}_out[{}*{}*{}];\n'.format(output_type,i,out_height,out_width,n_filt)
                        else: newline += '    {} layer{}_out[{}*{}*{}];\n'.format(output_type,i,in_height,in_width,n_filt)
                    if yamlConfig["IOType"] == "io_parallel": newline += '    #pragma HLS ARRAY_PARTITION variable=layer{}_out complete dim=0\n'.format(i)
                    if yamlConfig["IOType"] == "io_serial":   newline += '    #pragma HLS STREAM variable=layer{}_out depth=1\n'.format(i)

                #github Issue 53
                #Compute Dense layer
                #if layer_list[i-1]['activation'] == "linear" and layer_list[i-1]['class_name']=='Dense':
                #    newline += '    nnet::compute_layer<{}, {}, config{}>({}, {}, w{}, b{});\n'.format(input_type, output_type, i, input_object, output_object, i, i)
                #elif layer_list[i-1]['class_name']=='Dense':
                if layer_list[i-1]['class_name']=='Dense':
                    newline += '    {} logits{}[{}];\n'.format(output_type,i,n_out)
                    if yamlConfig["IOType"] == "io_parallel": newline += '    #pragma HLS ARRAY_PARTITION variable=logits{} complete dim=0\n'.format(i)
                    if yamlConfig["IOType"] == "io_serial":   newline += '    #pragma HLS STREAM variable=logits{} depth=1\n'.format(i)
                    
                    if layer_list[i-1]['n_part']==1 or yamlConfig["IOType"]=="io_serial":
                        # Use one layer if there's only 1 partition, or if we're using serial mode
                        newline += '    nnet::compute_layer<{}, {}, config{}>({}, logits{}, w{}, b{});\n'.format(input_type, output_type, i, input_object, i, i, i, i)
                    else:
                        # initialize arrays for sublayer outputs
                        newline += '    compute_layer{}({}, logits{});\n'.format(i, input_object, i)
                        sublayerline = 'void compute_layer{}({} {}[{}], {} logits{}[{}]) {{\n'.format(i,input_type, input_object, n_in, output_type, i, n_out)
                        sublayerline_h = 'void compute_layer{}({} {}[{}], {} logits{}[{}]);\n'.format(i,input_type, input_object, n_in, output_type, i, n_out)
                        sublayerlines_h.append(sublayerline_h)
                        for i_part in range(0, layer_list[i-1]['n_part']):
                            n_subout = layer_list[i-1]['n_subout'][i_part]
                            sublayerline += '    {} logits{}_{}[{}];\n'.format(output_type,i,i_part,n_subout)                        
                            if yamlConfig["IOType"] == "io_parallel": sublayerline += '    #pragma HLS ARRAY_PARTITION variable=logits{}_{} complete dim=0\n'.format(i,i_part)
                            if yamlConfig["IOType"] == "io_serial":   sublayerline += '    #pragma HLS STREAM variable=logits{}_{} depth=1\n'.format(i,i_part)

                        # initialize arrays for merged partial outputs 
                        for i_part in range(1, layer_list[i-1]['n_part']-1):
                            n_mergeout = sum([layer_list[i-1]['n_subout'][kk] for kk in range(0, i_part+1)])
                            sublayerline += '    {} logits{}_0to{}[{}];\n'.format(output_type,i,i_part,n_mergeout)                        
                            if yamlConfig["IOType"] == "io_parallel": sublayerline += '    #pragma HLS ARRAY_PARTITION variable=logits{}_0to{} complete dim=0\n'.format(i,i_part)
                            if yamlConfig["IOType"] == "io_serial":   sublayerline += '    #pragma HLS STREAM variable=logits{}_0to{} depth=1\n'.format(i,i_part)
                        # compute sublayer outputs
                        for i_part in range(0, layer_list[i-1]['n_part']):
                            sublayerline += '    nnet::compute_layer<{}, {}, config{}_{}>({}, logits{}_{}, w{}_{}, b{}_{});\n'.format(input_type, output_type, i, i_part, input_object, i, i_part, i, i_part, i, i_part)   

                        # merge sublayer outputs
                        for i_part in range(0, layer_list[i-1]['n_part']-1):
                            n_subout = layer_list[i-1]['n_subout'][i_part+1]
                            n_mergeout = sum([layer_list[i-1]['n_subout'][kk] for kk in range(0, i_part+1)])
                            if layer_list[i-1]['n_part']==2:
                                sublayerline += '    nnet::merge<{}, {}, {}>(logits{}_{}, logits{}_{}, logits{});\n'.format(output_type, n_mergeout, n_subout, i, i_part, i, i_part+1, i)
                            elif i_part==0: 
                                sublayerline += '    nnet::merge<{}, {}, {}>(logits{}_{}, logits{}_{}, logits{}_0to{});\n'.format(output_type, n_mergeout, n_subout, i, i_part, i, i_part+1, i, i_part+1)
                            elif i_part==layer_list[i-1]['n_part']-2:
                                sublayerline += '    nnet::merge<{}, {}, {}>(logits{}_0to{}, logits{}_{}, logits{});\n'.format(output_type, n_mergeout, n_subout, i, i_part, i, i_part+1, i)
                            else:
                                sublayerline += '    nnet::merge<{}, {}, {}>(logits{}_0to{}, logits{}_{}, logits{}_0to{});\n'.format(output_type, n_mergeout, n_subout, i, i_part, i, i_part+1, i, i_part+1)
                        sublayerline += '}\n'
                        sublayerlines.append(sublayerline)
                    
                elif layer_list[i-1]['class_name']=='Conv1D':
                    if i>1 and layer_list[i-2]['class_name']=='Conv1D':
                        newline += '    {} conv_layer{}_in[{}][{}];\n'.format(input_type,i,y_in,n_chan)
                        if yamlConfig["IOType"] == "io_parallel": newline += '    #pragma HLS ARRAY_PARTITION variable=conv_layer{}_in complete dim=0\n'.format(i)
                        if yamlConfig["IOType"] == "io_serial":   newline += '    #pragma HLS STREAM variable=conv_layer{}_in depth=1\n'.format(i)
                        newline += '    nnet::unflatten<{}, {}, {}>({}, conv_layer{}_in);\n'.format(input_type, y_in, n_chan, input_object, i)                              
                        newline += '    {} conv_layer{}_out[{}][{}];\n'.format(output_type,i,y_out,n_filt)
                        if yamlConfig["IOType"] == "io_parallel": newline += '    #pragma HLS ARRAY_PARTITION variable=conv_layer{}_out complete dim=0\n'.format(i)
                        if yamlConfig["IOType"] == "io_serial":   newline += '    #pragma HLS STREAM variable=conv_layer{}_out depth=1\n'.format(i)
                        newline += '    nnet::conv_1d<{}, {}, config{}>(conv_layer{}_in, conv_layer{}_out, w{}, b{});\n'.format(input_type, output_type, i, i, i, i, i, i)  
                    else:                        
                        newline += '    {} conv_layer{}_out[{}][{}];\n'.format(output_type,i,y_out,n_filt)
                        if yamlConfig["IOType"] == "io_parallel": newline += '    #pragma HLS ARRAY_PARTITION variable=conv_layer{}_out complete dim=0\n'.format(i)
                        if yamlConfig["IOType"] == "io_serial":   newline += '    #pragma HLS STREAM variable=conv_layer{}_out depth=1\n'.format(i)
                        newline += '    nnet::conv_1d<{}, {}, config{}>({}, conv_layer{}_out, w{}, b{});\n'.format(input_type, output_type, i, input_object, i, i, i, i)
                    newline += '    {} logits{}[{}*{}];\n'.format(output_type,i,y_out,n_filt)
                    if yamlConfig["IOType"] == "io_parallel": newline += '    #pragma HLS ARRAY_PARTITION variable=logits{} complete dim=0\n'.format(i)
                    if yamlConfig["IOType"] == "io_serial":   newline += '    #pragma HLS STREAM variable=logits{} complete depth=1\n'.format(i)
                    newline += '    nnet::flatten<{}, {}, {}>(conv_layer{}_out, logits{});\n'.format(input_type, y_out, n_filt, i, i)
                elif layer_list[i-1]['class_name']=='Conv2D':
                    if i>1 and (layer_list[i-2]['class_name']=='Conv2D' or layer_list[i-2]['class_name']=='BatchNormalization'):
                        newline += '    {} conv2d_layer{}_in[{}][{}][{}];\n'.format(input_type,i,in_height,in_width,n_chan)
                        if yamlConfig["IOType"] == "io_parallel": newline += '    #pragma HLS ARRAY_PARTITION variable=conv2d_layer{}_in complete dim=0\n'.format(i)
                        if yamlConfig["IOType"] == "io_serial":   newline += '    #pragma HLS STREAM variable=conv2d_layer{}_in depth=1\n'.format(i)
                        newline += '    nnet::unflatten<{}, {}, {}, {}>({}, conv2d_layer{}_in);\n'.format(input_type, in_height, in_width, n_chan, input_object, i)                              
                        newline += '    {} conv2d_layer{}_out[{}][{}][{}];\n'.format(output_type,i,out_height,out_width,n_filt)
                        if yamlConfig["IOType"] == "io_parallel": newline += '    #pragma HLS ARRAY_PARTITION variable=conv2d_layer{}_out complete dim=0\n'.format(i)
                        if yamlConfig["IOType"] == "io_serial":   newline += '    #pragma HLS STREAM variable=conv2d_layer{}_out depth=1\n'.format(i)
                        newline += '    nnet::conv_2d<{}, {}, config{}>(conv2d_layer{}_in, conv2d_layer{}_out, w{}, b{});\n'.format(input_type, output_type, i, i, i, i, i, i)  
                    else:                        
                        newline += '    {} conv2d_layer{}_out[{}][{}][{}];\n'.format(output_type,i,out_height,out_width,n_filt)
                        if yamlConfig["IOType"] == "io_parallel": newline += '    #pragma HLS ARRAY_PARTITION variable=conv2d_layer{}_out complete dim=0\n'.format(i)
                        if yamlConfig["IOType"] == "io_serial":   newline += '    #pragma HLS STREAM variable=conv2d_layer{}_out depth=1\n'.format(i)
                        newline += '    nnet::conv_2d<{}, {}, config{}>({}, conv2d_layer{}_out, w{}, b{});\n'.format(input_type, output_type, i, input_object, i, i, i, i)
                    newline += '    {} logits{}[{}*{}*{}];\n'.format(output_type,i,out_height,out_width,n_filt)
                    if yamlConfig["IOType"] == "io_parallel": newline += '    #pragma HLS ARRAY_PARTITION variable=logits{} complete dim=0\n'.format(i)
                    if yamlConfig["IOType"] == "io_serial":   newline += '    #pragma HLS STREAM variable=logits{} complete depth=1\n'.format(i)
                    newline += '    nnet::flatten<{}, {}, {}, {}>(conv2d_layer{}_out, logits{});\n'.format(output_type, out_height, out_width, n_filt, i, i)
                elif layer_list[i-1]['class_name'] == 'BatchNormalization' and is_dense:
                    newline += '    nnet::normalize<{}, {}, config{}>({}, {}, scale{}, beta{}, mean{});\n'.format(input_type, output_type, i, input_object, output_object, i, i, i)
                elif i==1 and layer_list[i-1]['class_name'] == 'BatchNormalization' and is_conv2d:
                    newline += '    {} logits{}[{}*{}*{}];\n'.format(output_type,i,in_height,in_width,n_filt)
                    if yamlConfig["IOType"] == "io_parallel": newline += '    #pragma HLS ARRAY_PARTITION variable=logits{} complete dim=0\n'.format(i)
                    if yamlConfig["IOType"] == "io_serial":   newline += '    #pragma HLS STREAM variable=logits{} complete depth=1\n'.format(i)
                    newline += '    nnet::flatten<{}, {}, {}, {}>({}, logits{});\n'.format(input_type, in_height, in_width, n_filt, input_object, i)
                    newline += '    nnet::normalize<{}, {}, config{}>(logits{}, {}, scale{}, beta{}, mean{});\n'.format(output_type, output_type, i, i, output_object, i, i, i)
                elif layer_list[i-1]['class_name'] == 'BatchNormalization' and is_conv2d:
                    newline += '    nnet::normalize<{}, {}, config{}>({}, {}, scale{}, beta{}, mean{});\n'.format(input_type, output_type, i, input_object, output_object, i, i, i)
                elif 'Pooling' in layer_list[i-1]['class_name']:
                    info = layer_list[i-1]['class_name'].split('Pooling')
                    d = int(info[1].split('D')[0]) # n dimensions
                    if d == 1:
                        newline += '    nnet::pooling1d<{}, config{}>({}, {});\n'.format(input_type, i, input_object, output_object)
                    elif d == 2:
                        # Unflatten if needed: if the last layer is activation or batchnorm
                        unflatten = layer_list[i-2]['class_name'] in activation_layers
                        unflatten |= 'activation' in layer_list[i-2].keys()
                        unflatten |= layer_list[i-2]['class_name'] == 'BatchNormalization'
                        if unflatten:
                            # Add the unflatten layer
                            inshape = ''.join('[{0}]'.format(dim) for dim in [in_height, in_width, n_filt])
                            newline += '    {} pool2d_layer{}_in{};\n'.format(input_type, i, inshape)
                            if yamlConfig["IOType"] == "io_parallel": newline += '    #pragma HLS ARRAY_PARTITION variable=pool2d_layer{}_in complete dim=0\n'.format(i)
                            if yamlConfig["IOType"] == "io_serial":   newline += '    #pragma HLS STREAM variable=pool2d_layer{}_in depth=1\n'.format(i)
                            newline += '    nnet::unflatten<{}, {}, {}, {}>({}, pool2d_layer{}_in);\n'.format(input_type, in_height, in_width, n_filt, input_object, i)                              
                            outshape = ''.join('[{0}]'.format(dim) for dim in [out_height, out_width, n_filt])
                            newline += '    {} pool2d_layer{}_out{};\n'.format(input_type, i, outshape)
                            if yamlConfig["IOType"] == "io_parallel": newline += '    #pragma HLS ARRAY_PARTITION variable=pool2d_layer{}_out complete dim=0\n'.format(i)
                            if yamlConfig["IOType"] == "io_serial":   newline += '    #pragma HLS STREAM variable=pool2d_layer{}_out depth=1\n'.format(i)
                            # Do the pooling layer
                            newline += '    nnet::pooling2d<{}, config{i}>(pool2d_layer{i}_in, pool2d_layer{i}_out);\n'.format(input_type, i=i)
                        else:
                            newline += '    nnet::pooling2d<{}, config{i}>({}, {});\n'.format(input_type, i, input_object, output_object)
                        # Flatten the pooling output
                        newline += '    nnet::flatten<{}, {}, {}, {}>(pool2d_layer{}_out, layer{}_out);\n'.format(input_type, out_height, out_width, n_filt, i, i)
                        
                #Activations
                if layer_list[i-1]['class_name'] in activation_layers or 'activation' in layer_list[i-1].keys():
                    if layer_list[i-1]['class_name'] not in activation_layers:
                        act_input_type = output_type
                        act_input_object = "logits" + str(i)
                    else:
                        act_input_type = input_type
                        act_input_object = input_object
                    
                    activation_name = layer_list[i-1]['activation']+'_config'+str(i)
                    activation_param = layer_list[i-1].get('activ_param')
                    if layer_list[i-1]['activation'] == "relu":
                        newline += '    nnet::relu<{}, {}, {}>({}, {});\n'.format(act_input_type, output_type, activation_name, act_input_object, output_object)
                    elif layer_list[i-1]['activation'] == "LeakyReLU":
                        newline += '    nnet::leaky_relu<{}, {}, {}>({}, {}, {});\n'.format(act_input_type, output_type, activation_name, act_input_object, activation_param, output_object)
                    elif layer_list[i-1]['activation'] == "ThresholdedReLU":
                        newline += '    nnet::thresholded_relu<{}, {}, {}>({}, {}, {});\n'.format(act_input_type, output_type, activation_name, act_input_object, activation_param, output_object)
                    elif layer_list[i-1]['activation'].lower() == "elu":
                        if activation_param:
                            newline += '    nnet::elu<{}, {}, {}>({}, {}, {});\n'.format(act_input_type, output_type, activation_name, act_input_object, activation_param, output_object)
                        else:
                            newline += '    nnet::elu<{}, {}, {}>({}, {});\n'.format(act_input_type, output_type, activation_name, act_input_object, output_object)
                    elif layer_list[i-1]['activation'] == "selu":
                        newline += '    nnet::selu<{}, {}, {}>({}, {});\n'.format(act_input_type, output_type, activation_name, act_input_object, output_object)
                    elif layer_list[i-1]['activation'] == "PReLU":
                        newline += '    nnet::prelu<{}, {}, {}>({}, a{}, {});\n'.format(act_input_type, output_type, activation_name, act_input_object, i, output_object)
                    elif layer_list[i-1]['activation'] == "softmax":
                        newline += '    nnet::softmax<{}, {}, {}>({}, {});\n'.format(act_input_type, output_type, activation_name, act_input_object, output_object)
                    elif layer_list[i-1]['activation'] == "sigmoid":
                        newline += '    nnet::sigmoid<{}, {}, {}>({}, {});\n'.format(act_input_type, output_type, activation_name, act_input_object, output_object)
                    elif layer_list[i-1]['activation'] == "hard_sigmoid":
                        newline += '    nnet::hard_sigmoid<{}, {}, {}>({}, {});\n'.format(act_input_type, output_type, activation_name, act_input_object, output_object)
                    elif layer_list[i-1]['activation'] == "tanh":
                        newline += '    nnet::tanh<{}, {}, {}>({}, {});\n'.format(act_input_type, output_type, activation_name, act_input_object, output_object)
                    elif layer_list[i-1]['activation'] == "linear": 
                        #github Issue 53
                        newline += '    nnet::linear<{}, {}, {}>({}, {});\n'.format(act_input_type, output_type, activation_name, act_input_object, output_object)
                    elif layer_list[i-1]['activation'] == "softsign":
                        newline += '    nnet::softsign<{}, {}, {}>({}, {});\n'.format(act_input_type, output_type, activation_name, act_input_object, output_object)
                    elif layer_list[i-1]['activation'] == "softplus":
                        newline += '    nnet::softplus<{}, {}, {}>({}, {});\n'.format(act_input_type, output_type, activation_name, act_input_object, output_object)
                    else:
                        raise Exception('ERROR: MISSING ACTIVATION')

                newline += '\n'

        #Just copy line
        else: 
            newline = line
        fout.write(newline)
    for sublayerline in sublayerlines:
        fout.write('\n')
        fout.write(sublayerline)
        fout.write('\n')
    f.close()
    fout.close()

    ###################
    ## parameters.h
    ###################

    f = open(os.path.join(filedir,'../hls-template/firmware/parameters.h'),'r')
    fout = open('{}/firmware/parameters.h'.format(yamlConfig['OutputDir']),'w')

    dense_config_template = """struct config{index} : nnet::layer_config {{
        static const unsigned n_in = {n_in};
        static const unsigned n_out = {n_out};
        static const unsigned io_type = nnet::{iotype};
        static const unsigned reuse_factor = {reuse};
        static const unsigned n_zeros = {nzeros};
        static const bool store_weights_in_bram = false;
        static const bool use_lowlatency = true;
        typedef accum_default_t accum_t;
        typedef bias_default_t bias_t;
        typedef weight_default_t weight_t;
        }};\n"""

    dense_sub_config_template = """struct config{index}_{i_part} : nnet::layer_config {{
        static const unsigned n_in = {n_in};
        static const unsigned n_out = {n_out};
        static const unsigned io_type = nnet::{iotype};
        static const unsigned reuse_factor = {reuse};
        static const unsigned n_zeros = {nzeros};
        static const bool store_weights_in_bram = false;
        static const bool use_lowlatency = true;
        typedef accum_default_t accum_t;
        typedef bias_default_t bias_t;
        typedef weight_default_t weight_t;
        }};\n"""

    batchnorm_config_template = """struct config{index} : nnet::batchnorm_config {{
        static const unsigned n_in = {n_in};
        static const unsigned n_filt = {n_filt};
        static const unsigned io_type = nnet::{iotype};
        static const unsigned reuse_factor = {reuse};
        static const bool store_weights_in_bram = false;
        typedef beta_default_t beta_t;
        typedef scale_default_t scale_t;
        typedef mean_default_t mean_t;
        }};\n"""

    conv_config_template = """struct config{index} : nnet::conv_config {{
        static const unsigned pad_left = {pad_left};
        static const unsigned pad_right = {pad_right};
        static const unsigned y_in = {y_in};
        static const unsigned n_chan = {n_chan};
        static const unsigned y_filt = {y_filt};
        static const unsigned n_filt = {n_filt};
        static const unsigned stride = {stride};
        static const unsigned y_out = {y_out};
        static const unsigned reuse_factor = {reuse};
        static const unsigned n_zeros = {nzeros};
        static const bool store_weights_in_bram = false;
        typedef accum_default_t accum_t;
        typedef bias_default_t bias_t;
        typedef weight_default_t weight_t;
        }};\n"""

    conv2d_config_template = """struct config{index} : nnet::conv2d_config {{
        static const unsigned pad_top = {pad_top};
        static const unsigned pad_bottom = {pad_bottom};
        static const unsigned pad_left = {pad_left};
        static const unsigned pad_right = {pad_right};
        static const unsigned in_height = {in_height};
        static const unsigned in_width = {in_width};
        static const unsigned n_chan = {n_chan};
        static const unsigned filt_height = {filt_height};
        static const unsigned filt_width = {filt_width};
        static const unsigned n_filt = {n_filt};
        static const unsigned stride_height = {stride_height};
        static const unsigned stride_width = {stride_width};
        static const unsigned out_height = {out_height};
        static const unsigned out_width = {out_width};
        static const unsigned reuse_factor = {reuse};
        static const unsigned n_zeros = {nzeros};
        static const bool store_weights_in_bram = false;
        typedef accum_default_t accum_t;
        typedef bias_default_t bias_t;
        typedef weight_default_t weight_t;
        }};\n"""
    
    

    activ_config_template = """struct {type}_config{index} : nnet::activ_config {{
        static const unsigned n_in = {n_in};
        static const unsigned table_size = 1024;
        static const unsigned io_type = nnet::{iotype};
        }};\n"""

    pooling1d_config_template = """struct config{index} : nnet::pooling1d_config {{
        static const unsigned n_in = {n_in};
        static const unsigned pool_size = {pool_size};
        static const unsigned n_out = {n_out};
        static const unsigned pad_left = {pad_left};
        static const unsigned pad_right = {pad_right};
        static const unsigned stride = {stride};
        static const nnet::Pool_Op pool_op = nnet::{Op};
    }};\n"""

    pooling2d_config_template = """struct config{index} : nnet::pooling2d_config {{
        static const unsigned in_height = {in_height};
        static const unsigned in_width = {in_width};
        static const unsigned n_filt = {n_filt};
        static const unsigned stride_height = {stride_height};
        static const unsigned stride_width = {stride_width};
        static const unsigned pool_height = {pool_height};
        static const unsigned pool_width = {pool_width};
        static const unsigned out_height = {out_height};
        static const unsigned out_width = {out_width};
        static const unsigned pad_top = {pad_top};
        static const unsigned pad_bottom = {pad_bottom};
        static const unsigned pad_left = {pad_left};
        static const unsigned pad_right = {pad_right};
        static const nnet::Pool_Op pool_op = nnet::{Op};
        static const unsigned reuse = {reuse};
    }};\n
    """

    for line in f.readlines():

        #Insert numbers
        if '//hls-fpga-machine-learning insert numbers' in line:
            newline = line
            newline += 'typedef {precision} accum_default_t;\n'.format(precision=yamlConfig["DefaultPrecision"])
            newline += 'typedef {precision} weight_default_t;\n'.format(precision=yamlConfig["DefaultPrecision"])
            newline += 'typedef {precision} bias_default_t;\n'.format(precision=yamlConfig["DefaultPrecision"])
            newline += 'typedef {precision} input_t;\n'.format(precision=yamlConfig["DefaultPrecision"])
            newline += 'typedef {precision} result_t;\n'.format(precision=yamlConfig["DefaultPrecision"])
            if do_batchnorm:
             newline += 'typedef {precision} beta_default_t;\n'.format(precision=yamlConfig["DefaultPrecision"])
             newline += 'typedef {precision} mean_default_t;\n'.format(precision=yamlConfig["DefaultPrecision"])
             newline += 'typedef {precision} scale_default_t;\n'.format(precision=yamlConfig["DefaultPrecision"])

            for i in range(1,len(layer_list)+1):
                if i==1 and layer_list[i-1]['class_name']=='Dense':
                    newline += '#define N_INPUTS {}\n'.format(layer_list[i-1]['n_in'])
                    newline += '#define N_LAYER_1 {}\n'.format(layer_list[i-1]['n_out'])
                elif i==1 and layer_list[i-1]['class_name']=='BatchNormalization' and is_dense:
                    newline += '#define N_INPUTS {}\n'.format(layer_list[i-1]['n_in'])
                    newline += '#define N_LAYER_1 {}\n'.format(layer_list[i-1]['n_out'])
                    newline += '#define N_FILT_1 {}\n'.format(layer_list[i-1]['n_filt'])
                elif i==1 and layer_list[i-1]['class_name']=='BatchNormalization' and is_conv2d:
                    newline += '#define N_INPUTS {}\n'.format(layer_list[i-1]['n_in'])
                    newline += '#define N_LAYER_{} {}\n'.format(i,layer_list[i-1]['n_out'])
                    newline += '#define IN_HEIGHT_{} {}\n'.format(i, layer_list[i-1]['in_height'])
                    newline += '#define IN_WIDTH_{} {}\n'.format(i, layer_list[i-1]['in_width'])
                    newline += '#define N_FILT_{} {}\n'.format(i, layer_list[i-1]['n_filt'])
                elif i==len(layer_list) and layer_list[i-1]['class_name']=='Dense':
                    newline += '#define N_OUTPUTS {}\n'.format(layer_list[i-1]['n_out'])
                elif i==len(layer_list) and layer_list[i-1]['class_name'] in activation_layers:
                    newline += '#define N_OUTPUTS {}\n'.format(layer_list[i-2]['n_out']) 
                elif i==len(layer_list) and layer_list[i-1]['class_name']=='BatchNormalization':
                    newline += '#define N_OUTPUTS {}\n'.format(layer_list[i-1]['n_out']) 
                    newline += '#define N_FILT_{} {}\n'.format(i-1, layer_list[i-1]['n_filt']) 
                elif layer_list[i-1]['class_name']=='Dense':
                    newline += '#define N_LAYER_{} {}\n'.format(i, layer_list[i-1]['n_out'])    
                elif is_dense and layer_list[i-1]['class_name']=='BatchNormalization':
                    newline += '#define N_LAYER_{} {}\n'.format(i, layer_list[i-1]['n_out'])  
                    newline += '#define N_FILT_{} {}\n'.format(i-1, layer_list[i-1]['n_filt']) 	
                elif layer_list[i-1]['class_name'] in activation_layers:        
                    newline += '#define N_LAYER_{} {}\n'.format(i, layer_list[i-2]['n_out'])
                elif layer_list[i-1]['class_name']=='Conv1D':
                    newline += '#define Y_INPUTS_{} {}\n'.format(i, layer_list[i-1]['y_in'])
                    newline += '#define N_CHAN_{} {}\n'.format(i, layer_list[i-1]['n_chan'])
                    newline += '#define Y_OUTPUTS_{} {}\n'.format(i, layer_list[i-1]['y_out'])
                    newline += '#define N_FILT_{} {}\n'.format(i, layer_list[i-1]['n_filt'])
                elif layer_list[i-1]['class_name']=='Conv2D':
                    newline += '#define IN_HEIGHT_{} {}\n'.format(i, layer_list[i-1]['in_height'])
                    newline += '#define IN_WIDTH_{} {}\n'.format(i, layer_list[i-1]['in_width'])
                    newline += '#define N_CHAN_{} {}\n'.format(i, layer_list[i-1]['n_chan'])
                    newline += '#define OUT_HEIGHT_{} {}\n'.format(i, layer_list[i-1]['out_height'])
                    newline += '#define OUT_WIDTH_{} {}\n'.format(i, layer_list[i-1]['out_width'])
                    newline += '#define N_FILT_{} {}\n'.format(i, layer_list[i-1]['n_filt'])
                elif layer_list[i-1]['class_name']=='BatchNormalization' and is_conv2d:
                    newline += '#define N_LAYER_{} {}\n'.format(i, layer_list[i-1]['n_out']) 
                    newline += '#define OUT_HEIGHT_{} {}\n'.format(i, layer_list[i-1]['in_height'])
                    newline += '#define OUT_WIDTH_{} {}\n'.format(i, layer_list[i-1]['in_width'])
                    newline += '#define N_FILT_{} {}\n'.format(i, layer_list[i-1]['n_filt']) 
                elif 'Pooling' in layer_list[i-1]['class_name']:
                    info = layer_list[i-1]['class_name'].split('Pooling')
                    d = int(info[1].split('D')[0])
                    op = info[0]
                    if d == 1:
                        newline += '#define Y_INPUTS_{} {}\n'.format(i, layer_list[i-1]['y_in'])
                        newline += '#define Y_OUTPUTS_{} {}\n'.format(i, layer_list[i-1]['y_out'])
                        newline += '#define POOL_SIZE_{} {}\n'.format(i, layer_list[i-1]['pool_size'])
                    elif d == 2:
                        newline += '#define IN_HEIGHT_{} {}\n'.format(i, layer_list[i-1]['in_height'])
                        newline += '#define IN_WIDTH_{} {}\n'.format(i, layer_list[i-1]['in_width'])
                        newline += '#define OUT_HEIGHT_{} {}\n'.format(i, layer_list[i-1]['out_height'])
                        newline += '#define OUT_WIDTH_{} {}\n'.format(i, layer_list[i-1]['out_width'])
                        newline += '#define POOL_HEIGHT_{} {}\n'.format(i, layer_list[i-1]['pool_height'])
                        newline += '#define POOL_WIDTH_{} {}\n'.format(i, layer_list[i-1]['pool_width'])
                        newline += '#define N_FILT_{} {}\n'.format(i, layer_list[i-1]['n_filt'])
                        newline += '#define N_LAYER_{} {}\n'.format(i, layer_list[i-1]['n_out'])


        elif '//hls-fpga-machine-learning insert layer-precision' in line:
            newline = line
            for i in range(1,len(layer_list)):
            #    if layer_list[i-1]['class_name']=='Dense':
            #        newline += 'typedef {precision} layer{index}_t;\n'.format(precision=yamlConfig["DefaultPrecision"], index=i)
                newline += 'typedef {precision} layer{index}_t;\n'.format(precision=yamlConfig["DefaultPrecision"], index=i)

        elif "//hls-fpga-machine-learning insert layer-config" in line:
            newline = line
            for i in range(1,len(layer_list)+1):
                if i==1 and (layer_list[i-1]['class_name']=='Dense' or layer_list[i-1]['class_name']=='BatchNormalization'):
                    layer_in_name = "N_INPUTS"
                    layer_out_name = "N_LAYER_1"                        
                    layer_n_filt_name = "N_FILT_1"
                elif i==1 and layer_list[i-1]['class_name']=='BatchNormalization' and is_conv2d:
                    layer_in_name = "IN_HEIGHT_{}*IN_WIDTH_{}*N_FILT_{}".format(i, i, i)
                    layer_out_name = "N_LAYER_1"       
                    layer_n_filt_name = "N_FILT_{}".format(i)
                elif i==1 and layer_list[i-1]['class_name']=='BatchNormalization' and is_dense:
                    layer_in_name = "N_INPUTS"
                    layer_out_name = "N_LAYER_1"       
                    layer_n_filt_name = "N_FILT_{}".format(i)
                elif is_dense and layer_list[i-1]['class_name']=='BatchNormalization':
                    layer_in_name = "N_LAYER_{}".format(i-1)
                    layer_out_name = "N_LAYER_{}".format(i)
                    layer_n_filt_name = "N_FILT_{}".format(i-1)
                elif i==len(layer_list) and layer_list[i-1]['class_name']=='Dense' and layer_list[i-2]['class_name']=='Conv1D':
                    layer_in_name = "Y_OUTPUTS_{}*N_FILT_{}".format(i-1, i-1)
                    layer_out_name = "N_OUTPUTS"
                elif i==len(layer_list) and layer_list[i-1]['class_name']=='Dense' and layer_list[i-2]['class_name']=='Conv2D':
                    layer_in_name = "OUT_HEIGHT_{}*OUT_WIDTH_{}*N_FILT_{}".format(i-1, i-1, i-1)
                    layer_out_name = "N_OUTPUTS"
                elif layer_list[i-1]['class_name']=='Dense' and layer_list[i-2]['class_name']=='Conv1D':
                    layer_in_name = "Y_OUTPUTS_{}*N_FILT_{}".format(i-1, i-1)
                    layer_out_name = "N_LAYER_{}".format(i)   
                elif layer_list[i-1]['class_name']=='Dense' and layer_list[i-2]['class_name']=='Conv2D':
                    layer_in_name = "OUT_HEIGHT_{}*OUT_WIDTH_{}*N_FILT_{}".format(i-1, i-1, i-1)
                    layer_out_name = "N_LAYER_{}".format(i)   
                elif i==len(layer_list) and (layer_list[i-1]['class_name']=='Dense' or (is_dense and layer_list[i-1]['class_name'] in activation_layers) or (is_dense and layer_list[i-1]['class_name']=='BatchNormalization')):
                    layer_in_name = "N_LAYER_{}".format(i-1)
                    layer_out_name = "N_OUTPUTS"               
                elif layer_list[i-1]['class_name']=='Dense' or (is_dense and layer_list[i-1]['class_name'] in activation_layers):
                    layer_in_name = "N_LAYER_{}".format(i-1)
                    layer_out_name = "N_LAYER_{}".format(i)
                elif layer_list[i-1]['class_name']=='Conv1D':
                    layer_y_in_name = "Y_INPUTS_{}".format(i)
                    layer_n_chan_name = "N_CHAN_{}".format(i)
                    layer_y_out_name = "Y_OUTPUTS_{}".format(i)
                    layer_n_filt_name = "N_FILT_{}".format(i)
                elif layer_list[i-1]['class_name']=='Conv2D': #or (is_conv2d and layer_list[i-1]['class_name']=='BatchNormalization'):
                    layer_in_height_name = "IN_HEIGHT_{}".format(i)
                    layer_in_width_name = "IN_WIDTH_{}".format(i)
                    layer_n_chan_name = "N_CHAN_{}".format(i)
                    layer_out_height_name = "OUT_HEIGHT_{}".format(i)
                    layer_out_width_name = "OUT_WIDTH_{}".format(i)
                    layer_n_filt_name = "N_FILT_{}".format(i)
                    layer_in_name = "N_LAYER_{}".format(i-1)
                elif is_conv2d and layer_list[i-1]['class_name']=='BatchNormalization':
                    layer_in_name = "OUT_HEIGHT_{}*OUT_WIDTH_{}*N_FILT_{}".format(i-1, i-1, i-1)
                    layer_out_name = "N_LAYER_{}".format(i)
                    layer_n_filt_name = "N_FILT_{}".format(i-1)
                elif 'Pooling' in layer_list[i-1]['class_name']:
                    info = layer_list[i-1]['class_name'].split('Pooling')
                    d = int(info[1].split('D')[0])
                    op = info[0]
                    if d == 1:
                        layer_y_in_name = "Y_INPUTS_{}".format(i)
                        layer_y_out_name = "Y_OUTPUTS_{}".format(i)
                        layer_n_filt_name = "N_FILT_{}".format(i)
                    elif d == 2:
                        layer_in_height_name = "IN_HEIGHT_{}".format(i)
                        layer_in_width_name = "IN_WIDTH_{}".format(i)
                        layer_out_height_name = "OUT_HEIGHT_{}".format(i)
                        layer_out_width_name = "OUT_WIDTH_{}".format(i)
                        layer_n_filt_name = "N_FILT_{}".format(i)
                        layer_in_name = "N_LAYER_{}".format(i-1)
                if layer_list[i-1]['class_name']=='Dense':
                    if layer_list[i-1]['n_part']==1:
                        newline += dense_config_template.format(index=str(i), 
                                                                n_in=layer_in_name, 
                                                                n_out=layer_out_name,
                                                                iotype=yamlConfig["IOType"],
                                                                reuse=yamlConfig["ReuseFactor"],
                                                                nzeros=layer_list[i-1]['weights_n_zeros'])
                    else:
                        for i_part in range(0, layer_list[i-1]['n_part']):
                            newline += dense_sub_config_template.format(index=str(i),
                                                                        i_part=i_part,
                                                                        n_in=layer_in_name,
                                                                        n_out=layer_list[i-1]['n_subout'][i_part],
                                                                        iotype=yamlConfig["IOType"],
                                                                        reuse=yamlConfig["ReuseFactor"],
                                                                        nzeros=layer_list[i-1]['weights_n_subzeros'][i_part])

                    newline += activ_config_template.format(type=layer_list[i-1]['activation'],
                                                                    index=str(i), 
                                                                    n_in=layer_out_name,
                                                                    iotype=yamlConfig["IOType"]) 
                elif layer_list[i-1]['class_name']=='BatchNormalization':
                    newline += batchnorm_config_template.format(index=str(i), 
                                                            n_in=layer_in_name, 
                                                            n_out=layer_out_name,
                                                            n_filt=layer_n_filt_name,
                                                            iotype=yamlConfig["IOType"],
                                                            reuse=yamlConfig["ReuseFactor"])
                elif layer_list[i-1]['class_name'] in activation_layers:	
                    newline += activ_config_template.format(type=layer_list[i-1]['activation'],
                                                                    index=str(i), 
                                                                    n_in=layer_out_name,
                                                                    iotype=yamlConfig["IOType"]) 
 
                elif layer_list[i-1]['class_name']=='Conv1D':
                    newline += conv_config_template.format(index=str(i), 
                                                            pad_left=layer_list[i-1]['pad_left'], 
                                                            pad_right=layer_list[i-1]['pad_right'],
                                                            y_in=layer_y_in_name,
                                                            n_chan=layer_n_chan_name,
                                                            y_out=layer_y_out_name,
                                                            n_filt=layer_n_filt_name,
                                                            y_filt=layer_list[i-1]['y_filt'],
                                                            stride=layer_list[i-1]['stride'],
                                                            iotype=yamlConfig["IOType"],
                                                            reuse=yamlConfig["ReuseFactor"],
                                                            nzeros=layer_list[i-1]['weights_n_zeros'])

                    newline += activ_config_template.format(type=layer_list[i-1]['activation'],
                                                                    index=str(i), 
                                                                    n_in='{}*{}'.format(layer_y_out_name,layer_n_filt_name),
                                                                    iotype=yamlConfig["IOType"]) 

                elif layer_list[i-1]['class_name']=='Conv2D':
                    newline += conv2d_config_template.format(index=str(i), 
                                                            pad_top=layer_list[i-1]['pad_top'], 
                                                            pad_bottom=layer_list[i-1]['pad_bottom'],
                                                            pad_left=layer_list[i-1]['pad_left'], 
                                                            pad_right=layer_list[i-1]['pad_right'],
                                                            in_height=layer_in_height_name,
                                                            in_width=layer_in_width_name,
                                                            n_chan=layer_n_chan_name,
                                                            out_height=layer_out_height_name,
                                                            out_width=layer_out_width_name,
                                                            n_filt=layer_n_filt_name,
                                                            filt_height=layer_list[i-1]['filt_height'],
                                                            filt_width=layer_list[i-1]['filt_width'],
                                                            stride_height=layer_list[i-1]['stride_height'],
                                                            stride_width=layer_list[i-1]['stride_width'],
                                                            iotype=yamlConfig["IOType"],
                                                            reuse=yamlConfig["ReuseFactor"],
                                                            nzeros=layer_list[i-1]['weights_n_zeros'])

                    newline += activ_config_template.format(type=layer_list[i-1]['activation'],
                                                                    index=str(i), 
                                                                    n_in='{}*{}*{}'.format(layer_out_height_name,layer_out_width_name,layer_n_filt_name),
                                                                    iotype=yamlConfig["IOType"]) 
                elif 'Pooling' in layer_list[i-1]['class_name']:
                    info = layer_list[i-1]['class_name'].split('Pooling')
                    d = int(info[1].split('D')[0])
                    op = info[0]
                    if d == 1:
                        newline += pooling1d_config_template.format(index=str(i),
                                                                    n_in=layer_n_in,
                                                                    n_out=layer_n_out,
                                                                    stride=layer_list[i-1]['stride'],
                                                                    pool_size=layer_list[i-1]['pool_size'],
                                                                    pad_left=layer_list[i-1]['pad_left'],
                                                                    pad_right=layer_list[i-1]['pad_right'],
                                                                    Op=op)
                    elif d == 2:
                        newline += pooling2d_config_template.format(index=str(i),
                                                                    in_height=layer_in_height_name,
                                                                    in_width=layer_in_width_name,
                                                                    out_height=layer_out_height_name,
                                                                    out_width=layer_out_width_name,
                                                                    n_filt=layer_n_filt_name,
                                                                    stride_height=layer_list[i-1]['stride_height'],
                                                                    stride_width=layer_list[i-1]['stride_width'],
                                                                    pool_height=layer_list[i-1]['pool_height'],
                                                                    pool_width=layer_list[i-1]['pool_width'],
                                                                    pad_left=layer_list[i-1]['pad_left'],
                                                                    pad_right=layer_list[i-1]['pad_right'],
                                                                    pad_top=layer_list[i-1]['pad_top'],
                                                                    pad_bottom=layer_list[i-1]['pad_bottom'],
                                                                    Op=op,
                                                                    reuse=yamlConfig["ReuseFactor"])

        else:
            newline = line
        fout.write(newline)
    f.close()
    fout.close()

    ###################
    ## test bench
    ###################

    f = open(os.path.join(filedir,'../hls-template/myproject_test.cpp'),'r')
    fout = open('{}/{}_test.cpp'.format(yamlConfig['OutputDir'], yamlConfig['ProjectName']),'w')

    for line in f.readlines():

        #Insert numbers
        if 'myproject' in line:
            newline = line.replace('myproject',yamlConfig['ProjectName'])
        elif '//hls-fpga-machine-learning insert data' in line and (layer_list[0]['class_name']=='Dense' or (is_dense and layer_list[0]['class_name']=='BatchNormalization')):
            newline = line
            newline += '  input_t  data_str[N_INPUTS] = {'
            for i in range(0,layer_list[0]['n_in']-1):
                newline += '0,'
            newline += '0};\n'
        elif '//hls-fpga-machine-learning insert data' in line and layer_list[0]['class_name']=='Conv1D':
            newline = line
            newline += '  input_t  data_str[Y_INPUTS_1][N_CHAN_1] = {'
            for i in range(0,layer_list[0]['y_in']*layer_list[0]['n_chan']-1):
                newline += '0,'
            newline += '0};\n'
        elif '//hls-fpga-machine-learning insert data' in line and layer_list[0]['class_name']=='Conv2D':
            newline = line
            newline += '  input_t  data_str[IN_HEIGHT_1][IN_WIDTH_1][N_CHAN_1] = {'
            for i in range(0,layer_list[0]['in_height']*layer_list[0]['in_width']*layer_list[0]['n_chan']-1):
                newline += '0,'
            newline += '0};\n'
        elif '//hls-fpga-machine-learning insert data' in line and is_conv2d and layer_list[0]['class_name']=='BatchNormalization':
            newline = line
            newline += '  input_t  data_str[IN_HEIGHT_1][IN_WIDTH_1][N_FILT_1] = {'
            for i in range(0,layer_list[0]['in_height']*layer_list[0]['in_width']*layer_list[0]['n_filt']-1):
                newline += '0,'
            newline += '0};\n'
        else:
            newline = line
        fout.write(newline)
    f.close()
    fout.close()


    #######################
    ## myproject.h
    #######################

    f = open(os.path.join(filedir,'../hls-template/firmware/myproject.h'),'r')
    fout = open('{}/firmware/{}.h'.format(yamlConfig['OutputDir'], yamlConfig['ProjectName']),'w')

    for line in f.readlines():

        if 'MYPROJECT' in line:
            newline = line.replace('MYPROJECT',format(yamlConfig['ProjectName'].upper()))
        elif 'void myproject(' in line:
            newline = 'void {}(\n'.format(yamlConfig['ProjectName'])
        elif 'input_t data[N_INPUTS]' in line and layer_list[0]['class_name']=='Conv1D':
            newline = line.replace('input_t data[N_INPUTS]','input_t data[Y_INPUTS_1][N_CHAN_1]')
        elif 'input_t data[N_INPUTS]' in line and layer_list[0]['class_name']=='Conv2D':
            newline = line.replace('input_t data[N_INPUTS]','input_t data[IN_HEIGHT_1][IN_WIDTH_1][N_CHAN_1]')
        elif 'input_t data[N_INPUTS]' in line and layer_list[0]['class_name']=='BatchNormalization' and is_conv2d:
            newline = line.replace('input_t data[N_INPUTS]','input_t data[IN_HEIGHT_1][IN_WIDTH_1][N_FILT_1]')
        elif '#endif' in line:
            for sublayerline_h in sublayerlines_h:
                fout.write(sublayerline_h)
            fout.write('\n#endif\n')
        else:
            newline = line
        fout.write(newline)

    f.close()
    fout.close()


    #######################
    ## build_prj.tcl
    #######################

    nnetdir = os.path.abspath(os.path.join(filedir, "../nnet_utils"))
    relpath = os.path.relpath(nnetdir, start=yamlConfig['OutputDir'])

    f = open(os.path.join(filedir,'../hls-template/build_prj.tcl'),'r')
    fout = open('{}/build_prj.tcl'.format(yamlConfig['OutputDir']),'w')

    for line in f.readlines():

        line = line.replace('myproject',yamlConfig['ProjectName'])
        line = line.replace('nnet_utils', relpath)

        if 'set_part {xc7vx690tffg1927-2}' in line:
            line = 'set_part {{{}}}\n'.format(yamlConfig['XilinxPart'])
        elif 'create_clock -period 5 -name default' in line:
            line = 'create_clock -period {} -name default\n'.format(yamlConfig['ClockPeriod'])

        fout.write(line)
    f.close()
    fout.close()


    ###################
    # Tarball output
    ###################
    with tarfile.open(yamlConfig['OutputDir'] + '.tar.gz', mode='w:gz') as archive:
        archive.add(yamlConfig['OutputDir'], recursive=True)



#######################################
## Config module
#######################################
def parse_config(config_file) :

    print("Loading configuration from", config_file)
    config = open(config_file, 'r')
    return yaml.load(config)

#######################################
## Print a bias or weight array to C++
#######################################
def print_array_to_cpp(name, a, odir, i_part = 0, n_part = 1, i_subout = 0, n_subout = 1):

    #put output in subdir for tarballing later
    #check if we're doing sublayer
    if n_part > 1:
        f=open("{}/firmware/weights/{}_{}.h".format(odir,name,i_part),"w")
        if len(a.shape)==2: # dense weight
            a = a[:,i_subout:i_subout+n_subout]
        elif len(a.shape)==1: # bias
            a = a[i_subout:i_subout+n_subout]
    else:
        f=open("{}/firmware/weights/{}.h".format(odir,name),"w")

    #count zeros
    zero_ctr = 0
    for x in np.nditer(a, order='C'):
        if x == 0: 
            zero_ctr += 1

    #meta data
    f.write("//Numpy array shape {}\n".format(a.shape))
    f.write("//Min {:.12f}\n".format(np.min(a)))
    f.write("//Max {:.12f}\n".format(np.max(a)))
    f.write("//Number of zeros {}\n".format(zero_ctr))
    f.write("\n")
    
    #c++ variable
    if re.match(r"^w\d*$", name) or re.match(r"^a\d*$", name):
        if n_part > 1:
            f.write("weight_default_t {}_{}".format(name,i_part))
        else:
            f.write("weight_default_t {}".format(name))
    elif re.match(r"^b\d*$", name):
        if n_part > 1:
            f.write("bias_default_t {}_{}".format(name,i_part))
        else:
            f.write("bias_default_t {}".format(name))
    elif re.match(r"^beta\d*$", name):
        f.write("beta_default_t {}".format(name))	
    elif re.match(r"^mean\d*$", name):
        f.write("mean_default_t {}".format(name))
    elif re.match(r"^scale\d*$", name):
        f.write("scale_default_t {}".format(name))
    else:
        raise Exception('ERROR: Unkown weights type')

    #hls doesn't like 3d arrays... unrolling to 1d
    #also doing for all (including 2d) arrays now
    f.write("[{}]".format(np.prod(a.shape)))
    f.write(" = {")
    
    #fill c++ array.  
    #not including internal brackets for multidimensional case
    i=0
    for x in np.nditer(a, order='C'):
        if i==0:
            f.write("%.12f" % x)
        else:
            f.write(", %.12f" % x)
        i=i+1
    f.write("};\n")
    f.close()

    return zero_ctr
