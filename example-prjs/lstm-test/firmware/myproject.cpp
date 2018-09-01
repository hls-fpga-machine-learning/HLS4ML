//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#include <iostream>

#include "parameters.h"
#include "myproject.h"

#include "nnet_layer.h"
#include "nnet_conv.h"
#include "nnet_conv2d.h"
#include "nnet_recursive.h"
#include "nnet_activation.h"

//hls-fpga-machine-learning insert weights
#include "weights/w1.h"
#include "weights/b1.h"
#include "weights/wr1.h"
#include "weights/w2.h"
#include "weights/b2.h"

void myproject(
		  input_t data[N_LOOP][N_INPUTS],
		  result_t res[N_OUTPUTS],
		  unsigned short &const_size_in,
		  unsigned short &const_size_out)
{

    //hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=data complete dim=0 
    #pragma HLS ARRAY_RESHAPE variable=res complete dim=0 
    #pragma HLS INTERFACE ap_vld port=data,res 
    #pragma HLS PIPELINE 


    const_size_in   = N_INPUTS*N_LOOP;
    const_size_out  = N_OUTPUTS;

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    static    layer1_t layer1_out[N_STATE_1];
    #pragma HLS ARRAY_PARTITION variable=layer1_out complete dim=0
    for(int ii = 0; ii < N_STATE_1; ii++) layer1_out[ii] = 0;
    for(int iloop = 0; iloop < N_LOOP; iloop++) { 
       nnet::lstm_static<input_t, input_t, config1,relu_config1,sigmoid_config1_lstm>(1,data[iloop],layer1_out,w1,wr1,b1);

    }

    result_t logits2[N_OUTPUTS];
    #pragma HLS ARRAY_PARTITION variable=logits2 complete dim=0
    compute_layer2(layer1_out, logits2);
    nnet::softmax<result_t, result_t, softmax_config2>(logits2, res);


}

void lstm_matrixmult_1 ( 
          input_t  data              [N_INPUTS],
          input_t  data_recurr       [N_STATE_1],
          layer1_t logits1     [N_STATE_1*4],
          layer1_t logitsnob1  [N_STATE_1*4],
          weight_default_t W1   [N_INPUTS*N_STATE_1*4],
          weight_default_t Wr1  [N_STATE_1*N_STATE_1*4],
          weight_default_t b1   [N_STATE_1*4]) { 
    nnet::matrixmult_Wb<input_t, layer1_t, 6,64, config1>(data      ,logits1   , W1,b1); 
    nnet::matrixmult_W<input_t, layer1_t, 16,64, config1>(data_recurr,logitsnob1, Wr1);
}


void compute_layer2(layer1_t layer1_out[N_LAYER_1], result_t logits2[N_OUTPUTS]) {
    result_t logits2_0[3];
    #pragma HLS ARRAY_PARTITION variable=logits2_0 complete dim=0
    result_t logits2_1[2];
    #pragma HLS ARRAY_PARTITION variable=logits2_1 complete dim=0
    nnet::compute_sublayer<layer1_t, result_t, config2_0>(layer1_out, logits2_0, w2, b2);
    nnet::compute_sublayer<layer1_t, result_t, config2_1>(layer1_out, logits2_1, w2, b2);
    nnet::merge<result_t, 3, 2>(logits2_0, logits2_1, logits2);
}

