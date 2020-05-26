
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

#ifndef NNET_RECURSIVE_H_
#define NNET_RECURSIVE_H_

#include "nnet_common.h"
#include "nnet_activation.h"
#include "nnet_dense.h"


namespace nnet {

struct rnn_config
{
    // Internal data type definitions
    typedef float state_t;
    typedef float U_t;  // State x Input
    typedef float W_t;  // State x State
    typedef float V_t;  // Output x State

    // Layer Sizes
    static const unsigned n_in = 10;
    static const unsigned n_out = 2;
    static const unsigned n_state = 2;
    static const unsigned activation_type = activ_relu;
    static const unsigned table_size = 1024;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
};

// Recusive Neural Network (RNN)
// Resources: 
//  - http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/
//  - https://github.com/pangolulu/rnn-from-scratch
// Notes:
//  - RNN naming conventions adopted from the above links
//      - newstate = activation(U*input + W*state)
//      - output   = V*newstate
//  - If softmax is needed on output, perform *outside* this operations
template<class data_T, class res_T, typename CONFIG_T, typename ACT_CONFIG_T>
void simple_rnn(
    data_T    data[CONFIG_T::n_in],
    res_T     res[CONFIG_T::n_out],
    typename CONFIG_T::state_t newstate[CONFIG_T::n_state],
    typename CONFIG_T::U_t     param_U[CONFIG_T::n_in]   [CONFIG_T::n_state],
    typename CONFIG_T::W_t     param_W[CONFIG_T::n_state][CONFIG_T::n_state],
    typename CONFIG_T::V_t     param_V[CONFIG_T::n_state][CONFIG_T::n_out])
{

    // Initialize the state variable -- will maintain state between function calls
    //static typename CONFIG_T::state_t newstate[CONFIG_T::n_state];
    std::cout << "Pre-State: [ "; for (int ii = 0; ii < CONFIG_T::n_state; ii++) std::cout << newstate[ii] << " "; std::cout << "]" << std::endl;

    // Operation: U*input
    data_T inputcache;
    typename CONFIG_T::state_t inputmult[CONFIG_T::n_in][CONFIG_T::n_state];
    typename CONFIG_T::state_t inputacc[CONFIG_T::n_state];
    InputProd1: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        inputcache = data[ii];
        InputProd2: for(int jj = 0; jj < CONFIG_T::n_state; jj++) {
            inputmult[ii][jj] = inputcache * param_U[ii][jj];
        }
    }
    for(int iacc = 0; iacc < CONFIG_T::n_state; iacc++) {
        inputacc[iacc] = 0;
    }
    InputAccum1: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        InputAccum2: for(int jj = 0; jj < CONFIG_T::n_state; jj++) {
            inputacc[jj] += inputmult[ii][jj];
        }
    }

    // Operation: W*state
    data_T statecache;
    typename CONFIG_T::state_t statemult[CONFIG_T::n_state][CONFIG_T::n_state];
    typename CONFIG_T::state_t stateacc[CONFIG_T::n_state];
    StateProd1: for(int ii = 0; ii < CONFIG_T::n_state; ii++) {
        statecache = newstate[ii];
        StateProd2: for(int jj = 0; jj < CONFIG_T::n_state; jj++) {
            statemult[ii][jj] = statecache * param_W[ii][jj];
        }
    }
    for(int iacc = 0; iacc < CONFIG_T::n_state; iacc++) {
        stateacc[iacc] = 0;
    }
    StateAccum1: for(int ii = 0; ii < CONFIG_T::n_state; ii++) {
        StateAccum2: for(int jj = 0; jj < CONFIG_T::n_state; jj++) {
            stateacc[jj] += statemult[ii][jj];
        }
    }

    // Operation: U*input + W*state
    typename CONFIG_T::state_t rawstate[CONFIG_T::n_state];
    for(int iacc = 0; iacc < CONFIG_T::n_state; iacc++) {
        rawstate[iacc] = inputacc[iacc] + stateacc[iacc];
    }

    std::cout << "Post-State: [ "; for (int ii = 0; ii < CONFIG_T::n_state; ii++) std::cout << rawstate[ii] << " "; std::cout << "]" << std::endl;

    // Run activation function
    if (CONFIG_T::activation_type == activ_relu){
        relu<typename CONFIG_T::state_t, typename CONFIG_T::state_t, ACT_CONFIG_T>(rawstate, newstate);
    }
    else if (CONFIG_T::activation_type == activ_sigmoid){
      sigmoid<typename CONFIG_T::state_t, typename CONFIG_T::state_t, ACT_CONFIG_T>(rawstate, newstate);
    }
    else if (CONFIG_T::activation_type == activ_tanh){
        tanh<typename CONFIG_T::state_t, typename CONFIG_T::state_t, ACT_CONFIG_T>(rawstate, newstate);
    }

    std::cout << "Activated State: [ "; for (int ii = 0; ii < CONFIG_T::n_state; ii++) std::cout << newstate[ii] << " "; std::cout << "]" << std::endl;

    // Operation: output = V*state
    data_T outputcache;
    typename CONFIG_T::state_t outputmult[CONFIG_T::n_state][CONFIG_T::n_out];
    OutputProd1: for(int ii = 0; ii < CONFIG_T::n_state; ii++) {
        outputcache = newstate[ii];
        OutputProd2: for(int jj = 0; jj < CONFIG_T::n_out; jj++) {
            outputmult[ii][jj] = outputcache * param_V[ii][jj];
        }
    }
    for(int iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
        res[iacc] = 0;
    }
    OutputAccum1: for(int ii = 0; ii < CONFIG_T::n_state; ii++) {
        OutputAccum2: for(int jj = 0; jj < CONFIG_T::n_out; jj++) {
            res[jj] += outputmult[ii][jj];
        }
    }

}
//Version of the RNN where Pipeline is impossible but resource usage is significantly smaller. Here the states are contained in a fixed variable that is maintained during recursive calls
template<class data_T, class res_T, typename CONFIG_T, typename ACT_CONFIG_T>
void simple_rnn_static(
    data_T    data[CONFIG_T::n_in],
    res_T     res[CONFIG_T::n_out],
    typename CONFIG_T::U_t     param_U[CONFIG_T::n_in]   [CONFIG_T::n_state],
    typename CONFIG_T::W_t     param_W[CONFIG_T::n_state][CONFIG_T::n_state],
    typename CONFIG_T::V_t     param_V[CONFIG_T::n_state][CONFIG_T::n_out])
{

    // Initialize the state variable -- will maintain state between function calls
    static typename CONFIG_T::state_t newstate[CONFIG_T::n_state];
    std::cout << "Pre-State: [ "; for (int ii = 0; ii < CONFIG_T::n_state; ii++) std::cout << newstate[ii] << " "; std::cout << "]" << std::endl;

    // Operation: U*input
    data_T inputcache;
    typename CONFIG_T::state_t inputmult[CONFIG_T::n_in][CONFIG_T::n_state];
    typename CONFIG_T::state_t inputacc[CONFIG_T::n_state];
    InputProd1: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        inputcache = data[ii];
        InputProd2: for(int jj = 0; jj < CONFIG_T::n_state; jj++) {
            inputmult[ii][jj] = inputcache * param_U[ii][jj];
        }
    }
    for(int iacc = 0; iacc < CONFIG_T::n_state; iacc++) {
        inputacc[iacc] = 0;
    }
    InputAccum1: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        InputAccum2: for(int jj = 0; jj < CONFIG_T::n_state; jj++) {
            inputacc[jj] += inputmult[ii][jj];
        }
    }

    // Operation: W*state
    data_T statecache;
    typename CONFIG_T::state_t statemult[CONFIG_T::n_state][CONFIG_T::n_state];
    typename CONFIG_T::state_t stateacc[CONFIG_T::n_state];
    StateProd1: for(int ii = 0; ii < CONFIG_T::n_state; ii++) {
        statecache = newstate[ii];
        StateProd2: for(int jj = 0; jj < CONFIG_T::n_state; jj++) {
            statemult[ii][jj] = statecache * param_W[ii][jj];
        }
    }
    for(int iacc = 0; iacc < CONFIG_T::n_state; iacc++) {
        stateacc[iacc] = 0;
    }
    StateAccum1: for(int ii = 0; ii < CONFIG_T::n_state; ii++) {
        StateAccum2: for(int jj = 0; jj < CONFIG_T::n_state; jj++) {
            stateacc[jj] += statemult[ii][jj];
        }
    }

    // Operation: U*input + W*state
    typename CONFIG_T::state_t rawstate[CONFIG_T::n_state];
    for(int iacc = 0; iacc < CONFIG_T::n_state; iacc++) {
        rawstate[iacc] = inputacc[iacc] + stateacc[iacc];
    }

    std::cout << "Post-State: [ "; for (int ii = 0; ii < CONFIG_T::n_state; ii++) std::cout << rawstate[ii] << " "; std::cout << "]" << std::endl;

    // Run activation function
    if (CONFIG_T::activation_type == activ_relu){
        relu<typename CONFIG_T::state_t, typename CONFIG_T::state_t, ACT_CONFIG_T>(rawstate, newstate);
    }
    else if (CONFIG_T::activation_type == activ_sigmoid){
        sigmoid<typename CONFIG_T::state_t, typename CONFIG_T::state_t, ACT_CONFIG_T>(rawstate, newstate);
    }
    else if (CONFIG_T::activation_type == activ_tanh){
        tanh<typename CONFIG_T::state_t, typename CONFIG_T::state_t, ACT_CONFIG_T>(rawstate, newstate);
    }

    std::cout << "Activated State: [ "; for (int ii = 0; ii < CONFIG_T::n_state; ii++) std::cout << newstate[ii] << " "; std::cout << "]" << std::endl;

    // Operation: output = V*state
    data_T outputcache;
    typename CONFIG_T::state_t outputmult[CONFIG_T::n_state][CONFIG_T::n_out];
    OutputProd1: for(int ii = 0; ii < CONFIG_T::n_state; ii++) {
        outputcache = newstate[ii];
        OutputProd2: for(int jj = 0; jj < CONFIG_T::n_out; jj++) {
            outputmult[ii][jj] = outputcache * param_V[ii][jj];
        }
    }
    for(int iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
        res[iacc] = 0;
    }
    OutputAccum1: for(int ii = 0; ii < CONFIG_T::n_state; ii++) {
        OutputAccum2: for(int jj = 0; jj < CONFIG_T::n_out; jj++) {
            res[jj] += outputmult[ii][jj];
        }
    }

}
struct lstm_config
{
    // Internal data type definitions
    typedef float weight_t;
    typedef float bias_t;

    // Layer Sizes
    static const unsigned n_in =  2;
    static const unsigned n_sequence = 2;
    static const unsigned n_out = 2;
    static const unsigned n_state = 2;
    static const unsigned n_4state = 8;
    static const unsigned activation_type = activ_relu;
    static const unsigned table_size = 1024;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
};
// Long Short term Memory NN (LSTM)
// Resources: 
// https://github.com/nicodjimenez/lstm/blob/master/lstm.py
// https://github.com/llSourcell/LSTM_Networks/blob/master/LSTM%20Demo.ipynb
// https://en.wikipedia.org/wiki/Long_short-term_memory
// Notes:
//  - LSTM naming conventions adopted from the above links
//      - s_newstate = activation(U*input + W*state)
//      - h_output   = activation(U*input + W*state)*activation(s_newstate)
//  - If softmax is needed on output, perform *outside* this operations
//  Originall had a version allows for the state in each layer to be saved, moved this to above (this requires are LARGE dense network at the end)
template<class data_T, class res_T, typename CONFIG_T>
  void lstm(int       index,
	    data_T    data      [CONFIG_T::n_in],
	    res_T     h_newstate[CONFIG_T::n_state],
            res_T     s_newstate[CONFIG_T::n_state],
	    typename CONFIG_T::weight_t     param  [CONFIG_T::n_state*4*CONFIG_T::n_in],
	    typename CONFIG_T::weight_t     param_r[CONFIG_T::n_state*4*CONFIG_T::n_state],
	    typename CONFIG_T::bias_t     param_b[CONFIG_T::n_state*4],
            typename CONFIG_T::bias_t     param_br[CONFIG_T::n_state*4]
	    ) {

  // Initialize the state variable -- will maintain state between function calls
  std::cout << "S Pre-State: [ "; for (int ii = 0; ii < CONFIG_T::n_state; ii++) std::cout << s_newstate[ii] << " "; std::cout << "]" << std::endl;
  std::cout << "H Pre-State: [ "; for (int ii = 0; ii < CONFIG_T::n_state; ii++) std::cout << h_newstate[ii] << " "; std::cout << "]" << std::endl;
  res_T tmpres      [CONFIG_T::n_state*4];
  res_T tmpres_state[CONFIG_T::n_state*4];
  res_T tmpres_ifo  [CONFIG_T::n_state*3]; //activated i,f,o matrices (keras notation)
  res_T tmpres_c    [CONFIG_T::n_state];   //activated c-matrix (keras notation)
  res_T inputacc_ifo[CONFIG_T::n_state*3]; //i,f,o matrices (keras notation)
  res_T inputacc_c  [CONFIG_T::n_state]; //c-matrix (keras notation)
  res_T s_actstate[CONFIG_T::n_state]; 
  #pragma HLS ARRAY_PARTITION variable=h_newstate   complete
  #pragma HLS ARRAY_PARTITION variable=s_newstate   complete
  #pragma HLS ARRAY_PARTITION variable=tmpres       complete
  #pragma HLS ARRAY_PARTITION variable=tmpres_state complete
  #pragma HLS ARRAY_PARTITION variable=tmpres_ifo   complete
  #pragma HLS ARRAY_PARTITION variable=tmpres_c     complete
  #pragma HLS ARRAY_PARTITION variable=inputacc_ifo complete
  #pragma HLS ARRAY_PARTITION variable=inputacc_c   complete
  #pragma HLS ARRAY_PARTITION variable=s_actstate   complete  

  nnet::dense_latency<data_T, res_T, typename CONFIG_T::mult_config1>(data      ,tmpres   , param,param_b);
  nnet::dense_latency<data_T, res_T, typename CONFIG_T::mult_config2>(h_newstate,tmpres_state, param_r, param_br);

  for(int iacc = 0; iacc < (3*CONFIG_T::n_state); iacc++) {
    int index = iacc; 
    if(iacc > 2*CONFIG_T::n_state-1) index = iacc + CONFIG_T::n_state;
    inputacc_ifo[iacc] = tmpres[index] + tmpres_state[index];
  } 
  for(int iacc = 0; iacc < (CONFIG_T::n_state); iacc++) {
    int index = iacc + CONFIG_T::n_state*2;
    inputacc_c[iacc] = tmpres[index] + tmpres_state[index];
  } 
  if(CONFIG_T::ACT_CONFIG_LSTM::activation_type == activ_relu){
    nnet::relu<data_T, typename CONFIG_T:: weight_t, typename CONFIG_T::ACT_CONFIG_LSTM>(inputacc_ifo, tmpres_ifo);
  }
  else if (CONFIG_T::ACT_CONFIG_LSTM::activation_type == activ_sigmoid){
    nnet::sigmoid<data_T, typename CONFIG_T::weight_t, typename CONFIG_T::ACT_CONFIG_LSTM>(inputacc_ifo, tmpres_ifo);
  }
  else if (CONFIG_T::ACT_CONFIG_LSTM::activation_type == activ_tanh){
    nnet::tanh<data_T, typename CONFIG_T::weight_t, typename CONFIG_T::ACT_CONFIG_LSTM>(inputacc_ifo, tmpres_ifo);
  }
  //Now for the confusion matrix
  if (CONFIG_T::ACT_CONFIG_T::activation_type == activ_relu){
    nnet::relu<data_T, typename CONFIG_T::weight_t, typename CONFIG_T::ACT_CONFIG_T>(inputacc_c, tmpres_c);
  }
  else if (CONFIG_T::ACT_CONFIG_T::activation_type == activ_sigmoid){
    nnet::sigmoid<data_T, typename CONFIG_T::weight_t, typename CONFIG_T::ACT_CONFIG_T>(inputacc_c, tmpres_c);
  }
  else if (CONFIG_T::ACT_CONFIG_T::activation_type == activ_tanh){
    nnet::tanh<data_T, typename CONFIG_T::weight_t, typename CONFIG_T::ACT_CONFIG_T>(inputacc_c, tmpres_c);
  }
  // Operation: s=g*i+sold*f (update state with buffer to avoid timing issues)
  for(int iacc = 0; iacc < (CONFIG_T::n_state); iacc++) {
#pragma HLS UNROLL
    s_newstate[iacc] =  tmpres_c[iacc]*tmpres_ifo[iacc] + s_newstate[iacc]*tmpres_ifo[iacc+(CONFIG_T::n_state)];
  }
  // Operation: h=act(s)*o
  if (CONFIG_T::ACT_CONFIG_T::activation_type == activ_relu){
    nnet::relu<data_T, typename CONFIG_T::weight_t, typename CONFIG_T::ACT_CONFIG_T>(s_newstate,s_actstate);
  }
  else if (CONFIG_T::ACT_CONFIG_T::activation_type == activ_sigmoid){
    nnet::sigmoid<data_T, typename CONFIG_T::weight_t, typename CONFIG_T::ACT_CONFIG_T>(s_newstate,s_actstate);
  }
  else if (CONFIG_T::ACT_CONFIG_T::activation_type == activ_tanh){
    nnet::tanh<data_T, typename CONFIG_T::weight_t, typename CONFIG_T::ACT_CONFIG_T>(s_newstate,s_actstate);
  }
  for(int iacc = 0; iacc < CONFIG_T::n_state; iacc++) {
    h_newstate[iacc] = tmpres_ifo[iacc+2*(CONFIG_T::n_state)]*s_actstate[iacc];
  }

  std::cout << "Post-State: s [ "; for (int ii = 0; ii < CONFIG_T::n_state; ii++) std::cout << s_newstate[ii] << " "; std::cout << "]" << std::endl;
  std::cout << "Post-State: h [ "; for (int ii = 0; ii < CONFIG_T::n_state; ii++) std::cout << h_newstate[ii] << " "; std::cout << "]" << std::endl;

}
template<class data_T, class res_T, typename CONFIG_T>

  void lstm_static(int       index,
		   data_T    data      [CONFIG_T::n_in],
		   res_T     h_newstate[CONFIG_T::n_state],
		   typename CONFIG_T::weight_t     param  [CONFIG_T::n_state*4*CONFIG_T::n_in],
		   typename CONFIG_T::weight_t     param_r[CONFIG_T::n_state*4*CONFIG_T::n_state],
		   typename CONFIG_T::bias_t     param_b[CONFIG_T::n_state*4],
                   typename CONFIG_T::bias_t     param_br[CONFIG_T::n_state*4]
		   ) {
  static res_T     s_newstate[CONFIG_T::n_state];
  // Initialize the state variable -- will maintain state between function calls
  res_T tmpres      [CONFIG_T::n_state*4];
  res_T tmpres_state[CONFIG_T::n_state*4];
  res_T tmpres_ifo  [CONFIG_T::n_state*3]; //activated i,f,o matrices (keras notation)
  res_T tmpres_c    [CONFIG_T::n_state];   //activated c-matrix (keras notation)
  res_T inputacc_ifo[CONFIG_T::n_state*3]; //i,f,o matrices (keras notation)
  res_T inputacc_c  [CONFIG_T::n_state]; //c-matrix (keras notation)
  res_T s_actstate[CONFIG_T::n_state]; 
  #pragma HLS ARRAY_PARTITION variable=h_newstate   complete
  #pragma HLS ARRAY_PARTITION variable=s_newstate   complete
  #pragma HLS ARRAY_PARTITION variable=tmpres       complete
  #pragma HLS ARRAY_PARTITION variable=tmpres_state complete
  #pragma HLS ARRAY_PARTITION variable=tmpres_ifo   complete
  #pragma HLS ARRAY_PARTITION variable=tmpres_c     complete
  #pragma HLS ARRAY_PARTITION variable=inputacc_ifo complete
  #pragma HLS ARRAY_PARTITION variable=inputacc_c   complete
  #pragma HLS ARRAY_PARTITION variable=s_actstate   complete  


  nnet::dense_latency<data_T, res_T, typename CONFIG_T::mult_config1>(data      ,tmpres   , param,param_b);
  nnet::dense_latency<data_T, res_T, typename CONFIG_T::mult_config2>(h_newstate,tmpres_state, param_r, param_br);

  for(int iacc = 0; iacc < (3*CONFIG_T::n_state); iacc++) {
#pragma HLS UNROLL
    int index = iacc; 
    if(iacc > 2*CONFIG_T::n_state-1) index = iacc + CONFIG_T::n_state;
    inputacc_ifo[iacc] = tmpres[index] + tmpres_state[index];
  } 
  for(int iacc = 0; iacc < (CONFIG_T::n_state); iacc++) {
#pragma HLS UNROLL
    int index = iacc + CONFIG_T::n_state*2;
    inputacc_c[iacc] = tmpres[index] + tmpres_state[index];
  } 


  if (CONFIG_T::ACT_CONFIG_LSTM::activation_type == activ_relu){
    nnet::relu<data_T, typename CONFIG_T:: weight_t, typename CONFIG_T::ACT_CONFIG_LSTM>(inputacc_ifo, tmpres_ifo);
  }
  else if (CONFIG_T::ACT_CONFIG_LSTM::activation_type == activ_sigmoid){
    nnet::sigmoid<data_T, typename CONFIG_T:: weight_t, typename CONFIG_T::ACT_CONFIG_LSTM>(inputacc_ifo, tmpres_ifo);
  }
  else if (CONFIG_T::ACT_CONFIG_LSTM::activation_type == activ_tanh){
    nnet::tanh<data_T, typename CONFIG_T:: weight_t, typename CONFIG_T::ACT_CONFIG_LSTM>(inputacc_ifo, tmpres_ifo);
  }
  //Now for the confusion matrix
  if (CONFIG_T::ACT_CONFIG_T::activation_type == activ_relu){
    nnet::relu<res_T, typename CONFIG_T:: weight_t, typename CONFIG_T::ACT_CONFIG_T>(inputacc_c, tmpres_c);
  }
  else if (CONFIG_T::ACT_CONFIG_T::activation_type == activ_sigmoid){
    nnet::sigmoid<res_T, typename CONFIG_T:: weight_t, typename CONFIG_T::ACT_CONFIG_T>(inputacc_c, tmpres_c);
  }
  else if (CONFIG_T::ACT_CONFIG_T::activation_type == activ_tanh){
    nnet::tanh<data_T, typename CONFIG_T:: weight_t, typename CONFIG_T::ACT_CONFIG_T>(inputacc_c, tmpres_c);
  }
  // Operation: s=g*i+sold*f (update state with buffer to avoid timing issues)
  for(int iacc = 0; iacc < (CONFIG_T::n_state); iacc++) {
    #pragma HLS UNROLL
    s_newstate[iacc] =  tmpres_c[iacc]*tmpres_ifo[iacc] + s_newstate[iacc]*tmpres_ifo[iacc+(CONFIG_T::n_state)];
  }
  // Operation: h=act(s)*o
  if (CONFIG_T::ACT_CONFIG_T::activation_type == activ_relu){
    nnet::relu<data_T, typename CONFIG_T:: weight_t, typename CONFIG_T::ACT_CONFIG_T>(s_newstate,s_actstate);
  }
  else if (CONFIG_T::ACT_CONFIG_T::activation_type == activ_sigmoid){
    nnet::sigmoid<data_T, typename CONFIG_T:: weight_t, typename CONFIG_T::ACT_CONFIG_T>(s_newstate,s_actstate);
  }
  else if (CONFIG_T::ACT_CONFIG_T::activation_type == activ_tanh){
    nnet::tanh<data_T, typename CONFIG_T:: weight_t, typename CONFIG_T::ACT_CONFIG_T>(s_newstate,s_actstate);
  }
  for(int iacc = 0; iacc < CONFIG_T::n_state; iacc++) {
#pragma HLS UNROLL
    h_newstate[iacc] = tmpres_ifo[iacc+2*(CONFIG_T::n_state)]*s_actstate[iacc];
    //h_newstate[iacc] = inputacc_ifo[iacc+2*(CONFIG_T::n_state)]*s_newstate[iacc];
  }
  std::cout << "Post-State: s [ "; for (int ii = 0; ii < CONFIG_T::n_state; ii++) std::cout << s_newstate[ii] << " "; std::cout << "]" << std::endl;
  std::cout << "Post-State: h [ "; for (int ii = 0; ii < CONFIG_T::n_state; ii++) std::cout << h_newstate[ii] << " "; std::cout << "]" << std::endl;
}

template<class data_T, class res_T, typename CONFIG_T>
  void lstm_loop(
      data_T    data      [CONFIG_T::n_sequence*CONFIG_T::n_in],
      res_T     layer2_out_addup[CONFIG_T::n_sequence_out*CONFIG_T::n_state],
      typename CONFIG_T::weight_t     param  [CONFIG_T::n_state*4*CONFIG_T::n_in],
      typename CONFIG_T::weight_t     param_r[CONFIG_T::n_state*4*CONFIG_T::n_state],
      typename CONFIG_T::bias_t     param_b[CONFIG_T::n_state*4],
            typename CONFIG_T::bias_t     param_br[CONFIG_T::n_state*4]
      ) {

    res_T     h_newstate[CONFIG_T::n_state];
    //res_T     s_newstate[CONFIG_T::n_state];
    data_T data_in[CONFIG_T::n_in];

    #pragma HLS ARRAY_PARTITION variable=h_newstate complete
    //#pragma HLS ARRAY_PARTITION variable=s_newstate complete

    //for(int ii = 0; ii < CONFIG_T::n_state; ii++) s_newstate[ii] = 0;
    for(int ii = 0; ii < CONFIG_T::n_state; ii++) h_newstate[ii] = 0;
    for(int iloop = 0; iloop < CONFIG_T::n_sequence; iloop++) {
      for(int j = 0; j < CONFIG_T::n_in; j++){data_in[j] =  data[j + iloop*CONFIG_T::n_in];}
      nnet::lstm_static<data_T, res_T, typename CONFIG_T::config2>(1,data_in,h_newstate, param,param_r,param_b, param_br);
      if (CONFIG_T::n_sequence_out > 1)
        for(int i=CONFIG_T::n_state*iloop, j=0; i<(CONFIG_T::n_state*(iloop+1)); i++,j++){
          layer2_out_addup[i] = h_newstate[j];
    }
    }
      if (CONFIG_T::n_sequence_out == 1)
        for(int i=0; i<(CONFIG_T::n_state); i++){
           layer2_out_addup[i] = h_newstate[i];
    }
}

// Struct for the GRU template

struct gru_config
{
    // Internal data type definitions
    typedef float weight_t;
    typedef float bias_t;
    typedef float accum_t;

    // Layer Sizes
    static const unsigned n_in =  2;
    static const unsigned n_out = 2;
    static const unsigned n_state = 2;
    static const unsigned n_sequence = 2;
    static const unsigned n_4state = 8;
    static const unsigned activation_type = activ_relu;
    static const unsigned table_size = 1024;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    static const unsigned n_zeros = 0;
};

template<class data_T, class res_T, typename CONFIG_T>
  void gru(
	    data_T    data      [CONFIG_T::n_in],
	    res_T     h_newstate[CONFIG_T::n_state],
	    typename CONFIG_T::weight_t     param   [CONFIG_T::n_state*3*CONFIG_T::n_in], // TODO - Check the layout of the param weights - refer page in copy!!
	    typename CONFIG_T::weight_t     param_zr[CONFIG_T::n_state*3*CONFIG_T::n_state],
	    typename CONFIG_T::bias_t       param_b [CONFIG_T::n_state*3],
      typename CONFIG_T::bias_t       param_br [CONFIG_T::n_state*3]
	    ) {
    // Initialize the state variable -- will maintain state between function calls
    std::cout << "I Input(Pr): [ "; for (int ii = 0; ii < CONFIG_T::n_in; ii++) std::cout << data[ii] << " "; std::cout << "]" << std::endl;
    std::cout << "H Pre-State: [ "; for (int ii = 0; ii < CONFIG_T::n_state; ii++) std::cout << h_newstate[ii] << " "; std::cout << "]" << std::endl;
    
    typename CONFIG_T::accum_t h_state_hin [CONFIG_T::n_state];
    typename CONFIG_T::accum_t tmpres      [CONFIG_T::n_state*3];
    typename CONFIG_T::accum_t tmpres_state_zr[CONFIG_T::n_state*3];
    typename CONFIG_T::accum_t tmpres_state_h [CONFIG_T::n_state];
    typename CONFIG_T::accum_t tmpres_zr   [CONFIG_T::n_state*2]; //activated i,f,o matrices (keras notation)
    typename CONFIG_T::accum_t tmpres_h    [CONFIG_T::n_state];   //activated c-matrix (keras notation)
    typename CONFIG_T::accum_t inputacc_zr [CONFIG_T::n_state*2]; //i,f,o matrices (keras notation)
    typename CONFIG_T::accum_t inputacc_h  [CONFIG_T::n_state]; //c-matrix (keras notation)

    #pragma HLS ARRAY_PARTITION variable=h_newstate      complete
    #pragma HLS ARRAY_PARTITION variable=h_newstate_hin  complete
    #pragma HLS ARRAY_PARTITION variable=tmpres          complete
    #pragma HLS ARRAY_PARTITION variable=tmpres_state_zr complete
    #pragma HLS ARRAY_PARTITION variable=tmpres_state_h  complete
    #pragma HLS ARRAY_PARTITION variable=tmpres_zr       complete
    #pragma HLS ARRAY_PARTITION variable=tmpres_h        complete
    #pragma HLS ARRAY_PARTITION variable=inputacc_zr     complete
    #pragma HLS ARRAY_PARTITION variable=inputacc_h      complete

    nnet::dense_latency<data_T, typename CONFIG_T::accum_t, typename CONFIG_T::mult_config1>(data, tmpres, param, param_b);
    nnet::dense_latency<res_T, typename CONFIG_T::accum_t, typename CONFIG_T::mult_config2>(h_newstate, tmpres_state_zr, param_zr, param_br);

    // Adding the individual vectors from the multiplication of tmpres = Wx*x(t); tmpres_state_zr = Wh*h(t-1); tmpres initialized with biases -- DONE
    for(int iacc = 0; iacc < (2*CONFIG_T::n_state); iacc++) {
      int index = iacc; 
      inputacc_zr[iacc] = tmpres[index] + tmpres_state_zr[index];
    }

    // Activation function Sub layer -- START
    if (CONFIG_T::ACT_CONFIG_GRU::activation_type == activ_relu){
      relu<typename CONFIG_T::accum_t, typename CONFIG_T::weight_t, typename CONFIG_T::ACT_CONFIG_GRU>(inputacc_zr, tmpres_zr);
    }
    else if (CONFIG_T::ACT_CONFIG_GRU::activation_type == activ_sigmoid){
      sigmoid<typename CONFIG_T::accum_t, typename CONFIG_T::weight_t, typename CONFIG_T::ACT_CONFIG_GRU>(inputacc_zr, tmpres_zr);
    }
    else if (CONFIG_T::ACT_CONFIG_GRU::activation_type == activ_tanh){
      tanh<typename CONFIG_T::accum_t, typename CONFIG_T::weight_t, typename CONFIG_T::ACT_CONFIG_GRU>(inputacc_zr, tmpres_zr);
    }
    // Activation function Sub layer -- END

    // Hadamrd product of r(t) = inputacc_zr[2*n_state:n_state] and h(t-1) = h_newstate
    for(int iacc = 0; iacc < (CONFIG_T::n_state); iacc++) {
      #pragma HLS UNROLL
      tmpres_state_h[iacc] = tmpres_zr[iacc+(CONFIG_T::n_state)]*tmpres_state_zr[iacc + (2*CONFIG_T::n_state)];
    }

    //Assuming reset_after is false
    for(int iacc = 0; iacc < (CONFIG_T::n_state); iacc++) {
      #pragma HLS UNROLL
      int index = iacc + CONFIG_T::n_state*2;
      inputacc_h[iacc] =  tmpres[index] + tmpres_state_h[iacc];
    }

    //Now run the activation on this guy
    if (CONFIG_T::ACT_CONFIG_T::activation_type == activ_relu){
      relu<typename CONFIG_T::accum_t, typename CONFIG_T::weight_t, typename CONFIG_T::ACT_CONFIG_T>(inputacc_h, tmpres_h);
    }
    else if (CONFIG_T::ACT_CONFIG_T::activation_type == activ_sigmoid){
      sigmoid<typename CONFIG_T::accum_t, typename CONFIG_T::weight_t, typename CONFIG_T::ACT_CONFIG_T>(inputacc_h, tmpres_h);
    }
    else if (CONFIG_T::ACT_CONFIG_T::activation_type == activ_tanh){
      tanh<typename CONFIG_T::accum_t, typename CONFIG_T::weight_t, typename CONFIG_T::ACT_CONFIG_T>(inputacc_h, tmpres_h);
    }
    
    //Mix the stat with the previous state
    for(int iacc = 0; iacc < (CONFIG_T::n_state); iacc++) {
    #pragma HLS UNROLL
      h_newstate[iacc] =  (res_T)(tmpres_h[iacc]*(1-tmpres_zr[iacc]) + h_newstate[iacc]*tmpres_zr[iacc]);
    }
    std::cout << "Post-State: h [ "; for (int ii = 0; ii < CONFIG_T::n_state; ii++) std::cout << h_newstate[ii] << " "; std::cout << "]" << std::endl;  
}

template<class data_T, class res_T, typename CONFIG_T>
  void gru_static(int       index,
		  data_T    data      [CONFIG_T::n_in],
	    res_T     h_newstate[CONFIG_T::n_state],
	    typename CONFIG_T::weight_t     param   [CONFIG_T::n_state*3*CONFIG_T::n_in],
	    typename CONFIG_T::weight_t     param_zr[CONFIG_T::n_state*3*CONFIG_T::n_state],
	    typename CONFIG_T::bias_t       param_b [CONFIG_T::n_state*3],
      typename CONFIG_T::bias_t       param_br [CONFIG_T::n_state*3]
	    ) {
    // Initialize the state variable -- will maintain state between function calls
    
    static res_T h_state[CONFIG_T::n_state];
    typename CONFIG_T::accum_t h_state_hin [CONFIG_T::n_state];
    typename CONFIG_T::accum_t tmpres      [CONFIG_T::n_state*3];
    typename CONFIG_T::accum_t tmpres_state_zr[CONFIG_T::n_state*3];
    typename CONFIG_T::accum_t tmpres_state_h [CONFIG_T::n_state];
    typename CONFIG_T::accum_t tmpres_zr   [CONFIG_T::n_state*2]; //activated i,f,o matrices (keras notation)
    typename CONFIG_T::accum_t tmpres_h    [CONFIG_T::n_state];   //activated c-matrix (keras notation)
    typename CONFIG_T::accum_t inputacc_zr [CONFIG_T::n_state*2]; //i,f,o matrices (keras notation)
    typename CONFIG_T::accum_t inputacc_h  [CONFIG_T::n_state]; //c-matrix (keras notation)

    #pragma HLS ARRAY_PARTITION variable=h_state         complete
    #pragma HLS ARRAY_PARTITION variable=h_newstate      complete
    #pragma HLS ARRAY_PARTITION variable=h_newstate_hin  complete
    #pragma HLS ARRAY_PARTITION variable=tmpres          complete
    #pragma HLS ARRAY_PARTITION variable=tmpres_state_zr complete
    #pragma HLS ARRAY_PARTITION variable=tmpres_state_h  complete
    #pragma HLS ARRAY_PARTITION variable=tmpres_zr       complete
    #pragma HLS ARRAY_PARTITION variable=tmpres_h        complete
    #pragma HLS ARRAY_PARTITION variable=inputacc_zr     complete
    #pragma HLS ARRAY_PARTITION variable=inputacc_h      complete

    std::cout << "I Input(Pr): [ "; for (int ii = 0; ii < CONFIG_T::n_in; ii++) std::cout << data[ii] << " "; std::cout << "]" << std::endl;
    std::cout << "H Pre-State: [ "; for (int ii = 0; ii < CONFIG_T::n_state; ii++) std::cout << h_state[ii] << " "; std::cout << "]" << std::endl;

    nnet::dense_latency<data_T, typename CONFIG_T::accum_t, typename CONFIG_T::mult_config1>(data, tmpres, param, param_b);
    nnet::dense_latency<res_T, typename CONFIG_T::accum_t, typename CONFIG_T::mult_config2>(h_state, tmpres_state_zr, param_zr, param_br);

    // Adding the individual vectors from the multiplication of tmpres = Wx*x(t); tmpres_state_zr = Wh*h(t-1); tmpres initialized with biases -- DONE
    for(int iacc = 0; iacc < (2*CONFIG_T::n_state); iacc++) {
      int index = iacc; 
      inputacc_zr[iacc] = tmpres[index] + tmpres_state_zr[index];
    }

    // Activation function Sub layer -- START
    if (CONFIG_T::ACT_CONFIG_GRU::activation_type == activ_relu){
      relu<typename CONFIG_T::accum_t, typename CONFIG_T::weight_t, typename CONFIG_T::ACT_CONFIG_GRU>(inputacc_zr, tmpres_zr);
    }
    else if (CONFIG_T::ACT_CONFIG_GRU::activation_type == activ_sigmoid){
      sigmoid<typename CONFIG_T::accum_t, typename CONFIG_T::weight_t, typename CONFIG_T::ACT_CONFIG_GRU>(inputacc_zr, tmpres_zr);
    }
    else if (CONFIG_T::ACT_CONFIG_GRU::activation_type == activ_tanh){
      tanh<typename CONFIG_T::accum_t, typename CONFIG_T::weight_t, typename CONFIG_T::ACT_CONFIG_GRU>(inputacc_zr, tmpres_zr);
    }
    // Activation function Sub layer -- END

    // Hadamrd product of r(t) = inputacc_zr[2*n_state:n_state] and h(t-1) = h_newstate
    for(int iacc = 0; iacc < (CONFIG_T::n_state); iacc++) {
      #pragma HLS UNROLL
      tmpres_state_h[iacc] = tmpres_zr[iacc+(CONFIG_T::n_state)]*tmpres_state_zr[iacc + (2*CONFIG_T::n_state)];
    }

    //Assuming reset_after is false
    for(int iacc = 0; iacc < (CONFIG_T::n_state); iacc++) {
      #pragma HLS UNROLL
      int index = iacc + CONFIG_T::n_state*2;
      inputacc_h[iacc] =  tmpres[index] + tmpres_state_h[iacc];
    }

    //Now run the activation on this guy
    if (CONFIG_T::ACT_CONFIG_T::activation_type == activ_relu){
      relu<typename CONFIG_T::accum_t, typename CONFIG_T::weight_t, typename CONFIG_T::ACT_CONFIG_T>(inputacc_h, tmpres_h);
    }
    else if (CONFIG_T::ACT_CONFIG_T::activation_type == activ_sigmoid){
      sigmoid<typename CONFIG_T::accum_t, typename CONFIG_T::weight_t, typename CONFIG_T::ACT_CONFIG_T>(inputacc_h, tmpres_h);
    }
    else if (CONFIG_T::ACT_CONFIG_T::activation_type == activ_tanh){
      tanh<typename CONFIG_T::accum_t, typename CONFIG_T::weight_t, typename CONFIG_T::ACT_CONFIG_T>(inputacc_h, tmpres_h);
    }
    
    //Mix the stat with the previous state
    for(int iacc = 0; iacc < (CONFIG_T::n_state); iacc++) {
    #pragma HLS UNROLL
      h_state[iacc] =  (res_T)(tmpres_h[iacc]*(1-tmpres_zr[iacc]) + h_state[iacc]*tmpres_zr[iacc]);
      h_newstate[iacc] = h_state[iacc];
    }
    std::cout << "Post-State: h [ "; for (int ii = 0; ii < CONFIG_T::n_state; ii++) std::cout << h_newstate[ii] << " "; std::cout << "]" << std::endl;
}

template<class data_T, class res_T, typename CONFIG_T>
  void gru_loop(
	    data_T    data      [CONFIG_T::n_sequence*CONFIG_T::n_in],
            res_T     h_newstate[CONFIG_T::n_sequence_out*CONFIG_T::n_state],
	    typename CONFIG_T::weight_t     param   [CONFIG_T::n_state*3*CONFIG_T::n_in],
	    typename CONFIG_T::weight_t     param_zr[CONFIG_T::n_state*3*CONFIG_T::n_state],
	    typename CONFIG_T::bias_t       param_b [CONFIG_T::n_state*3],
      typename CONFIG_T::bias_t       param_br [CONFIG_T::n_state*3]
	    ) {

      res_T h_state[CONFIG_T::n_state];
      data_T data_in[CONFIG_T::n_in];

      #pragma HLS ARRAY_PARTITION variable=h_state complete
      #pragma HLS ARRAY_PARTITION variable=data_in complete

      for(int ii = 0; ii < CONFIG_T::n_state; ii++) h_state[ii] = 0;
      for(int iloop = 0; iloop < CONFIG_T::n_sequence; iloop++) {
        for(int j = 0; j < CONFIG_T::n_in; j++){data_in[j] = data[j + iloop*CONFIG_T::n_in];}
        nnet::gru<data_T, res_T, CONFIG_T>(data_in,h_state,param,param_zr,param_b, param_br);
        if (CONFIG_T::n_sequence_out > 1)
          for(int i=CONFIG_T::n_state*iloop, j=0; i<(CONFIG_T::n_state*(iloop+1)); i++,j++){
            h_newstate[i] = h_state[j];
          }
      }
      if (CONFIG_T::n_sequence_out == 1)
        for(int i=0; i<(CONFIG_T::n_state); i++){
          h_newstate[i] = h_state[i];
        }
    }

}//end namespace

#endif
