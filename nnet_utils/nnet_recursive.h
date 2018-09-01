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
template<class data_T, class res_T, typename CONFIG_T, typename ACT_CONFIG_T, typename ACT_CONFIG_LSTM>
  void lstm(int       index,
	    data_T    data      [CONFIG_T::n_in],
	    res_T     h_newstate[CONFIG_T::n_state],
	    res_T     s_newstate[CONFIG_T::n_state],
	    typename CONFIG_T::weight_t     param  [CONFIG_T::n_state*4*CONFIG_T::n_in],
	    typename CONFIG_T::weight_t     param_r[CONFIG_T::n_state*4*CONFIG_T::n_state],
	    typename CONFIG_T::bias_t     param_b[CONFIG_T::n_state*4]
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
  lstm_matrixmult_1(data,h_newstate,tmpres,tmpres_state,param,param_r,param_b);
  /*
  } else if(index == 2) { 
    lstm_matrixmult_2(data,h_newstate,tmpres,tmpres_state,param,param_r,param_b);
  } else { 
    lstm_matrixmult_3(data,h_newstate,tmpres,tmpres_state,param,param_r,param_b);
    }*/
  for(int iacc = 0; iacc < (3*CONFIG_T::n_state); iacc++) {
    int index = iacc; 
    if(iacc > 2*CONFIG_T::n_state-1) index = iacc + CONFIG_T::n_state;
    inputacc_ifo[iacc] = tmpres[index] + tmpres_state[index];
  } 
  for(int iacc = 0; iacc < (CONFIG_T::n_state); iacc++) {
    int index = iacc + CONFIG_T::n_state*2;
    inputacc_c[iacc] = tmpres[index] + tmpres_state[index];
  } 
  if (ACT_CONFIG_LSTM::activation_type == activ_relu){
    relu<res_T, typename CONFIG_T::weight_t, ACT_CONFIG_LSTM>(inputacc_ifo, tmpres_ifo);
  }
  else if (ACT_CONFIG_LSTM::activation_type == activ_sigmoid){
    sigmoid<res_T, typename CONFIG_T::weight_t, ACT_CONFIG_LSTM>(inputacc_ifo, tmpres_ifo);
  }
  else if (ACT_CONFIG_LSTM::activation_type == activ_tanh){
    tanh<res_T, typename CONFIG_T::weight_t, ACT_CONFIG_LSTM>(inputacc_ifo, tmpres_ifo);
  }
  //Now for the confusion matrix
  if (ACT_CONFIG_T::activation_type == activ_relu){
    relu<res_T, typename CONFIG_T::weight_t, ACT_CONFIG_T>(inputacc_c, tmpres_c);
  }
  else if (ACT_CONFIG_T::activation_type == activ_sigmoid){
    sigmoid<res_T, typename CONFIG_T::weight_t, ACT_CONFIG_T>(inputacc_c, tmpres_c);
  }
  else if (ACT_CONFIG_T::activation_type == activ_tanh){
    tanh<res_T, typename CONFIG_T::weight_t, ACT_CONFIG_T>(inputacc_c, tmpres_c);
  }
  // Operation: s=g*i+sold*f (update state with buffer to avoid timing issues)
  for(int iacc = 0; iacc < (CONFIG_T::n_state); iacc++) {
#pragma HLS UNROLL
    s_newstate[iacc] =  tmpres_c[iacc]*tmpres_ifo[iacc] + s_newstate[iacc]*tmpres_ifo[iacc+(CONFIG_T::n_state)];
  }
  // Operation: h=act(s)*o
  if (ACT_CONFIG_T::activation_type == activ_relu){
    relu<res_T, typename CONFIG_T::weight_t, ACT_CONFIG_T>(s_newstate,s_actstate);
  }
  else if (ACT_CONFIG_T::activation_type == activ_sigmoid){
    sigmoid<res_T, typename CONFIG_T::weight_t, ACT_CONFIG_T>(s_newstate,s_actstate);
  }
  else if (ACT_CONFIG_T::activation_type == activ_tanh){
    tanh<res_T, typename CONFIG_T::weight_t, ACT_CONFIG_T>(s_newstate,s_actstate);
  }
  for(int iacc = 0; iacc < CONFIG_T::n_state; iacc++) {
    h_newstate[iacc] = tmpres_ifo[iacc+2*(CONFIG_T::n_state)]*s_actstate[iacc];
  }
  std::cout << "Post-State: s [ "; for (int ii = 0; ii < CONFIG_T::n_state; ii++) std::cout << s_newstate[ii] << " "; std::cout << "]" << std::endl;
  std::cout << "Post-State: h [ "; for (int ii = 0; ii < CONFIG_T::n_state; ii++) std::cout << h_newstate[ii] << " "; std::cout << "]" << std::endl;
}
template<class data_T, class res_T, typename CONFIG_T, typename ACT_CONFIG_T, typename ACT_CONFIG_LSTM>
  void lstm_static(int       index,
		   data_T    data      [CONFIG_T::n_in],
		   res_T     h_newstate[CONFIG_T::n_state],
		   typename CONFIG_T::weight_t     param  [CONFIG_T::n_state*4*CONFIG_T::n_in],
		   typename CONFIG_T::weight_t     param_r[CONFIG_T::n_state*4*CONFIG_T::n_state],
		   typename CONFIG_T::bias_t     param_b[CONFIG_T::n_state*4]
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
  lstm_matrixmult_1(data,h_newstate,tmpres,tmpres_state,param,param_r,param_b);
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
  if (ACT_CONFIG_LSTM::activation_type == activ_relu){
    relu<res_T, typename CONFIG_T::weight_t, ACT_CONFIG_LSTM>(inputacc_ifo, tmpres_ifo);
  }
  else if (ACT_CONFIG_LSTM::activation_type == activ_sigmoid){
    sigmoid<res_T, typename CONFIG_T::weight_t, ACT_CONFIG_LSTM>(inputacc_ifo, tmpres_ifo);
  }
  else if (ACT_CONFIG_LSTM::activation_type == activ_tanh){
    tanh<res_T, typename CONFIG_T::weight_t, ACT_CONFIG_LSTM>(inputacc_ifo, tmpres_ifo);
  }
  //Now for the confusion matrix
  if (ACT_CONFIG_T::activation_type == activ_relu){
    relu<res_T, typename CONFIG_T::weight_t, ACT_CONFIG_T>(inputacc_c, tmpres_c);
  }
  else if (ACT_CONFIG_T::activation_type == activ_sigmoid){
    sigmoid<res_T, typename CONFIG_T::weight_t, ACT_CONFIG_T>(inputacc_c, tmpres_c);
  }
  else if (ACT_CONFIG_T::activation_type == activ_tanh){
    tanh<res_T, typename CONFIG_T::weight_t, ACT_CONFIG_T>(inputacc_c, tmpres_c);
  }
  // Operation: s=g*i+sold*f (update state with buffer to avoid timing issues)
  for(int iacc = 0; iacc < (CONFIG_T::n_state); iacc++) {
    #pragma HLS UNROLL
    s_newstate[iacc] =  tmpres_c[iacc]*tmpres_ifo[iacc] + s_newstate[iacc]*tmpres_ifo[iacc+(CONFIG_T::n_state)];
  }
  // Operation: h=act(s)*o
  if (ACT_CONFIG_T::activation_type == activ_relu){
    relu<res_T, typename CONFIG_T::weight_t, ACT_CONFIG_T>(s_newstate,s_actstate);
  }
  else if (ACT_CONFIG_T::activation_type == activ_sigmoid){
    sigmoid<res_T, typename CONFIG_T::weight_t, ACT_CONFIG_T>(s_newstate,s_actstate);
  }
  else if (ACT_CONFIG_T::activation_type == activ_tanh){
    tanh<res_T, typename CONFIG_T::weight_t, ACT_CONFIG_T>(s_newstate,s_actstate);
  }
  for(int iacc = 0; iacc < CONFIG_T::n_state; iacc++) {
#pragma HLS UNROLL
    h_newstate[iacc] = tmpres_ifo[iacc+2*(CONFIG_T::n_state)]*s_actstate[iacc];
    //h_newstate[iacc] = inputacc_ifo[iacc+2*(CONFIG_T::n_state)]*s_newstate[iacc];
  }
  std::cout << "Post-State: s [ "; for (int ii = 0; ii < CONFIG_T::n_state; ii++) std::cout << s_newstate[ii] << " "; std::cout << "]" << std::endl;
  std::cout << "Post-State: h [ "; for (int ii = 0; ii < CONFIG_T::n_state; ii++) std::cout << h_newstate[ii] << " "; std::cout << "]" << std::endl;
}
struct gru_config
{
    // Internal data type definitions
    typedef float weight_t;
    typedef float bias_t;

    // Layer Sizes
    static const unsigned n_in =  2;
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
template<class data_T, class res_T, typename CONFIG_T, typename ACT_CONFIG_T, typename ACT_CONFIG_LSTM>
  void gru(int       index,
	    data_T    data      [CONFIG_T::n_in],
	    res_T     h_newstate[CONFIG_T::n_state],
	    typename CONFIG_T::weight_t     param   [CONFIG_T::n_state*3*CONFIG_T::n_in],
	    typename CONFIG_T::weight_t     param_zr[CONFIG_T::n_state*2*CONFIG_T::n_state],
            typename CONFIG_T::weight_t     param_h [CONFIG_T::n_state*1*CONFIG_T::n_state],
	    typename CONFIG_T::bias_t       param_b [CONFIG_T::n_state*3]
	    ) {
  // Initialize the state variable -- will maintain state between function calls
  std::cout << "H Pre-State: [ "; for (int ii = 0; ii < CONFIG_T::n_state; ii++) std::cout << h_newstate[ii] << " "; std::cout << "]" << std::endl;
  res_T h_state_hin [CONFIG_T::n_state];
  res_T tmpres      [CONFIG_T::n_state*3];
  res_T tmpres_state_zr[CONFIG_T::n_state*2];
  res_T tmpres_state_h [CONFIG_T::n_state];
  res_T tmpres_zr   [CONFIG_T::n_state*2]; //activated i,f,o matrices (keras notation)
  res_T tmpres_h    [CONFIG_T::n_state];   //activated c-matrix (keras notation)
  res_T inputacc_zr [CONFIG_T::n_state*2]; //i,f,o matrices (keras notation)
  res_T inputacc_h  [CONFIG_T::n_state]; //c-matrix (keras notation)
  #pragma HLS ARRAY_PARTITION variable=h_newstate      complete
  #pragma HLS ARRAY_PARTITION variable=h_newstate_hin  complete
  #pragma HLS ARRAY_PARTITION variable=tmpres          complete
  #pragma HLS ARRAY_PARTITION variable=tmpres_state_zr complete
  #pragma HLS ARRAY_PARTITION variable=tmpres_state_h  complete
  #pragma HLS ARRAY_PARTITION variable=tmpres_zr       complete
  #pragma HLS ARRAY_PARTITION variable=tmpres_h        complete
  #pragma HLS ARRAY_PARTITION variable=inputacc_zr     complete
  #pragma HLS ARRAY_PARTITION variable=inputacc_h      complete
  gru_matrixmult_1_0(data,h_newstate,tmpres,tmpres_state_zr,param,param_zr,param_b);
  for(int iacc = 0; iacc < (2*CONFIG_T::n_state); iacc++) {
    int index = iacc; 
    inputacc_zr[iacc] = tmpres[index] + tmpres_state_zr[index];
  } 
  if (ACT_CONFIG_LSTM::activation_type == activ_relu){
    relu<res_T, typename CONFIG_T::weight_t, ACT_CONFIG_LSTM>(inputacc_zr, tmpres_zr);
  }
  else if (ACT_CONFIG_LSTM::activation_type == activ_sigmoid){
    sigmoid<res_T, typename CONFIG_T::weight_t, ACT_CONFIG_LSTM>(inputacc_zr, tmpres_zr);
  }
  else if (ACT_CONFIG_LSTM::activation_type == activ_tanh){
    tanh<res_T, typename CONFIG_T::weight_t, ACT_CONFIG_LSTM>(inputacc_zr, tmpres_zr);
  }
  for(int iacc = 0; iacc < (CONFIG_T::n_state); iacc++) {
    h_state_hin[iacc] =  tmpres_zr[iacc+(CONFIG_T::n_state)]*h_newstate[iacc];
  }
  gru_matrixmult_1_1(h_state_hin,tmpres_state_h,param_h);
  //Assuming reset_after is false
  for(int iacc = 0; iacc < (CONFIG_T::n_state); iacc++) {
    int index = iacc + CONFIG_T::n_state*2;
    inputacc_h[iacc] =  tmpres[index] + tmpres_state_h[iacc];
  }
  //Now run the activation on this guy
  if (ACT_CONFIG_T::activation_type == activ_relu){
    relu<res_T, typename CONFIG_T::weight_t, ACT_CONFIG_T>(inputacc_h, tmpres_h);
  }
  else if (ACT_CONFIG_T::activation_type == activ_sigmoid){
    sigmoid<res_T, typename CONFIG_T::weight_t, ACT_CONFIG_T>(inputacc_h, tmpres_h);
  }
  else if (ACT_CONFIG_T::activation_type == activ_tanh){
    tanh<res_T, typename CONFIG_T::weight_t, ACT_CONFIG_T>(inputacc_h, tmpres_h);
  }
  //Mix the stat with the previous state
  for(int iacc = 0; iacc < (CONFIG_T::n_state); iacc++) {
#pragma HLS UNROLL
    h_newstate[iacc] =  tmpres_h[iacc]*(1-tmpres_zr[iacc]) + h_newstate[iacc]*tmpres_zr[iacc];
  }
  std::cout << "Post-State: h [ "; for (int ii = 0; ii < CONFIG_T::n_state; ii++) std::cout << h_newstate[ii] << " "; std::cout << "]" << std::endl;
}

template<class data_T, class res_T, typename CONFIG_T, typename ACT_CONFIG_T, typename ACT_CONFIG_LSTM>
  void gru_static(int       index,
		  data_T    data      [CONFIG_T::n_in],
		  res_T     h_newstate[CONFIG_T::n_state],
		  typename CONFIG_T::weight_t     param   [CONFIG_T::n_state*3*CONFIG_T::n_in],
		  typename CONFIG_T::weight_t     param_zr[CONFIG_T::n_state*2*CONFIG_T::n_state],
		  typename CONFIG_T::weight_t     param_h [CONFIG_T::n_state*1*CONFIG_T::n_state],
		  typename CONFIG_T::bias_t       param_b [CONFIG_T::n_state*3]
		  ) {
  static res_T h_state[CONFIG_T::n_state];
  // Initialize the state variable -- will maintain state between function calls
  std::cout << "H Pre-State: [ "; for (int ii = 0; ii < CONFIG_T::n_state; ii++) std::cout << h_newstate[ii] << " "; std::cout << "]" << std::endl;
  res_T h_state_hin [CONFIG_T::n_state];
  res_T tmpres      [CONFIG_T::n_state*3];
  res_T tmpres_state_zr[CONFIG_T::n_state*2];
  res_T tmpres_state_h [CONFIG_T::n_state];
  res_T tmpres_zr   [CONFIG_T::n_state*2]; //activated i,f,o matrices (keras notation)
  res_T tmpres_h    [CONFIG_T::n_state];   //activated c-matrix (keras notation)
  res_T inputacc_zr [CONFIG_T::n_state*2]; //i,f,o matrices (keras notation)
  res_T inputacc_h  [CONFIG_T::n_state]; //c-matrix (keras notation)
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
  gru_matrixmult_1_0(data,h_state,tmpres,tmpres_state_zr,param,param_zr,param_b);
  for(int iacc = 0; iacc < (2*CONFIG_T::n_state); iacc++) {
    int index = iacc; 
    inputacc_zr[iacc] = tmpres[index] + tmpres_state_zr[index];
  } 
  if (ACT_CONFIG_LSTM::activation_type == activ_relu){
    relu<res_T, typename CONFIG_T::weight_t, ACT_CONFIG_LSTM>(inputacc_zr, tmpres_zr);
  }
  else if (ACT_CONFIG_LSTM::activation_type == activ_sigmoid){
    sigmoid<res_T, typename CONFIG_T::weight_t, ACT_CONFIG_LSTM>(inputacc_zr, tmpres_zr);
  }
  else if (ACT_CONFIG_LSTM::activation_type == activ_tanh){
    tanh<res_T, typename CONFIG_T::weight_t, ACT_CONFIG_LSTM>(inputacc_zr, tmpres_zr);
  }
  for(int iacc = 0; iacc < (CONFIG_T::n_state); iacc++) {
    h_state_hin[iacc] =  tmpres_zr[iacc+(CONFIG_T::n_state)]*h_state[iacc];
  }
  gru_matrixmult_1_1(h_state_hin,tmpres_state_h,param_h);
  //Assuming reset_after is false
  for(int iacc = 0; iacc < (CONFIG_T::n_state); iacc++) {
    int index = iacc + CONFIG_T::n_state*2;
    inputacc_h[iacc] =  tmpres[index] + tmpres_state_h[iacc];
  }
  //Now run the activation on this guy
  if (ACT_CONFIG_T::activation_type == activ_relu){
    relu<res_T, typename CONFIG_T::weight_t, ACT_CONFIG_T>(inputacc_h, tmpres_h);
  }
  else if (ACT_CONFIG_T::activation_type == activ_sigmoid){
    sigmoid<res_T, typename CONFIG_T::weight_t, ACT_CONFIG_T>(inputacc_h, tmpres_h);
  }
  else if (ACT_CONFIG_T::activation_type == activ_tanh){
    tanh<res_T, typename CONFIG_T::weight_t, ACT_CONFIG_T>(inputacc_h, tmpres_h);
  }
  std::cout << "H output [ "; for (int ii = 0; ii < CONFIG_T::n_state; ii++) std::cout << tmpres_h[ii] << " "; std::cout << "]" << std::endl;
  //Mix the stat with the previous state
  for(int iacc = 0; iacc < (CONFIG_T::n_state); iacc++) {
#pragma HLS UNROLL
    h_state[iacc]       =  tmpres_h[iacc]*(1-tmpres_zr[iacc]) + h_state[iacc]*tmpres_zr[iacc];
  }
  for(int iacc = 0; iacc < (CONFIG_T::n_state); iacc++) {
#pragma HLS UNROLL
    h_newstate[iacc]    =  h_static[iacc];
  }
  std::cout << "Post-State: h [ "; for (int ii = 0; ii < CONFIG_T::n_state; ii++) std::cout << h_newstate[ii] << " "; std::cout << "]" << std::endl;
}


}//end namespace

#endif
