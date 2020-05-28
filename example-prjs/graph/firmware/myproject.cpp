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

#include "myproject.h"
#include "parameters.h"

void myproject(
	       input_t    X[N_NODES][N_FEATURES],
	       ap_uint<1>  Ri[N_NODES][N_EDGES],
	       ap_uint<1>  Ro[N_NODES][N_EDGES],
	       result_t   e[N_EDGES][1],
	       unsigned short &const_size_in,
	       unsigned short &const_size_out)
{
  
  //hls-fpga-machine-learning insert IO
#pragma HLS ARRAY_RESHAPE variable=X complete dim=0 
#pragma HLS ARRAY_RESHAPE variable=Ri complete dim=0 
#pragma HLS ARRAY_RESHAPE variable=Ro complete dim=0 
#pragma HLS ARRAY_RESHAPE variable=e complete dim=0 
#pragma HLS INTERFACE ap_vld port=X,Ri,Ro,e
#pragma HLS PIPELINE 
  
  
  const_size_in   = N_NODES*N_FEATURES+2*N_NODES*N_EDGES;
  const_size_out  = N_EDGES*1;
  
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        //hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<model_default_t, 12>(w1, "w1.txt");
        nnet::load_weights_from_txt<model_default_t, 4>(b1, "b1.txt");
        nnet::load_weights_from_txt<model_default_t, 56>(w2, "w2.txt");
        nnet::load_weights_from_txt<model_default_t, 4>(b2, "b2.txt");
        nnet::load_weights_from_txt<model_default_t, 4>(w3, "w3.txt");
        nnet::load_weights_from_txt<model_default_t, 1>(b3, "b3.txt");
        nnet::load_weights_from_txt<model_default_t, 84>(w4, "w4.txt");
        nnet::load_weights_from_txt<model_default_t, 4>(b4, "b4.txt");
        nnet::load_weights_from_txt<model_default_t, 16>(w5, "w5.txt");
        nnet::load_weights_from_txt<model_default_t, 4>(b5, "b5.txt");

        loaded_weights = true;
    }
#endif

  // ****************************************
  // NETWORK INSTANTIATION
  // ****************************************
  
  //hls-fpga-machine-learning insert layers
  /*
  std::cout << "X = " << std::endl;
  for(int i=0; i<N_NODES; i++){
    for(int j=0; j<N_FEATURES; j++){
      std::cout << X[i][j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  */
  // input network
  input_t H_logits[N_NODES][N_HIDDEN_FEATURES];
  #pragma HLS ARRAY_PARTITION variable=H_logits complete dim=0
  nnet::dense_batch<input_t, input_t, dense_config1>(X, H_logits, w1, b1);
  /*
  std::cout << "H_logits = " << std::endl;
  for(int i=0; i<N_NODES; i++){
    for(int j=0; j<N_HIDDEN_FEATURES; j++){
      std::cout << H_logits[i][j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  */
  input_t H[N_NODES][N_HIDDEN_FEATURES];
  #pragma HLS ARRAY_PARTITION variable=H complete dim=0
  nnet::tanh_batch<input_t, input_t, tanh_config1>(H_logits, H);
  /*
  std::cout << "H = " << std::endl;
  for(int i=0; i<N_NODES; i++){
    for(int j=0; j<N_HIDDEN_FEATURES; j++){
      std::cout << H[i][j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  */
  input_t HX[N_NODES][N_FEATURES+N_HIDDEN_FEATURES];
  #pragma HLS ARRAY_PARTITION variable=HX complete dim=0
  nnet::merge2d<input_t, N_NODES, N_HIDDEN_FEATURES, N_FEATURES>(H, X, HX);
  
  // edge network 
  input_t B[N_EDGES][2*(N_FEATURES+N_HIDDEN_FEATURES)];
  #pragma HLS ARRAY_PARTITION variable=B complete dim=0
  nnet::compute_edge_net_features<input_t, input_t, graph_config1>(HX, Ri, Ro, B);

  input_t layer2_logits[N_EDGES][N_HIDDEN_FEATURES];
  #pragma HLS ARRAY_PARTITION variable=layer2_logits complete dim=0
  nnet::dense_batch<input_t, input_t, dense_config2>(B, layer2_logits, w2, b2);

  input_t layer2_out[N_EDGES][N_HIDDEN_FEATURES];
  #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
  nnet::tanh_batch<input_t, input_t, tanh_config2>(layer2_logits, layer2_out);

  input_t e_logits_temp[N_EDGES][1];
  #pragma HLS ARRAY_PARTITION variable=e_logits_temp complete dim=0
  nnet::dense_batch<input_t, input_t, dense_config3>(layer2_out, e_logits_temp, w3, b3);

  input_t e_temp[N_EDGES][1];
  #pragma HLS ARRAY_PARTITION variable=e_temp complete dim=0
  nnet::sigmoid_batch<input_t, input_t, sigmoid_config1>(e_logits_temp, e_temp);
  /*
  std::cout << "e_temp = " << std::endl;
  for(int i=0; i<N_EDGES; i++){
    std::cout << e_temp[i][0] << " ";
  }
  std::cout << std::endl;
  std::cout << std::endl;
  */
  // node network
  input_t M[N_NODES][3*(N_FEATURES+N_HIDDEN_FEATURES)];
  #pragma HLS ARRAY_PARTITION variable=M complete dim=0
  nnet::compute_node_net_features<input_t, input_t, graph_config1>(HX, e_temp, Ri, Ro, M);
  
  input_t layer4_logits[N_NODES][N_HIDDEN_FEATURES];
  #pragma HLS ARRAY_PARTITION variable=layer4_logits complete dim=0
  nnet::dense_batch<input_t, input_t, dense_config4>(M, layer4_logits, w4, b4);
  
  input_t layer4_out[N_NODES][N_HIDDEN_FEATURES];
  nnet::tanh_batch<input_t, input_t, tanh_config3>(layer4_logits, layer4_out);

  nnet::dense_batch<input_t, input_t, dense_config5>(layer4_out, H_logits, w5, b5);    

  nnet::tanh_batch<input_t, input_t, tanh_config4>(H_logits, H);
  /*
  std::cout << "H = " << std::endl;
  for(int i=0; i<N_NODES; i++){
    for(int j=0; j<N_HIDDEN_FEATURES; j++){
      std::cout << H[i][j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  */
  nnet::merge2d<input_t, N_NODES, N_HIDDEN_FEATURES, N_FEATURES>(H, X, HX);

  // edge network 
  nnet::compute_edge_net_features<input_t, input_t, graph_config1>(HX, Ri, Ro, B);

  input_t layer6_logits[N_EDGES][N_HIDDEN_FEATURES];
  #pragma HLS ARRAY_PARTITION variable=layer6_logits complete dim=0
  nnet::dense_batch<input_t, input_t, dense_config2>(B, layer6_logits, w2, b2);    

  input_t layer6_out[N_EDGES][N_HIDDEN_FEATURES];
  #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
  nnet::tanh_batch<input_t, input_t, tanh_config2>(layer6_logits, layer6_out);

  input_t e_logits[N_EDGES][1];
  #pragma HLS ARRAY_PARTITION variable=e_logits complete dim=0
  nnet::dense_batch<input_t, input_t, dense_config3>(layer6_out, e_logits, w3, b3);

  nnet::sigmoid_batch<input_t, input_t, sigmoid_config1>(e_logits, e);  
}
