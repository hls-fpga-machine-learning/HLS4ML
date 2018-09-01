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

#ifndef NNET_LAYER_H_
#define NNET_LAYER_H_

#include "nnet_common.h"
#include "hls_stream.h"
#include <math.h>

namespace nnet {

struct layer_config
{
    // Internal data type definitions
    typedef float bias_t;
    typedef float weight_t;
    typedef float accum_t;

    // Layer Sizes
    static const unsigned n_in = 10;
    static const unsigned n_out = 10;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    static const unsigned n_zeros = 0;
    // partitioning arrays cyclically to go with roll factors?
};

 template<class data_T, class res_T, typename CONFIG_T>
void compute_layer(
    data_T    data[CONFIG_T::n_in],
    res_T     res[CONFIG_T::n_out],
    typename CONFIG_T::weight_t  weights[CONFIG_T::n_in*CONFIG_T::n_out],
    typename CONFIG_T::bias_t    biases[CONFIG_T::n_out])
{
    data_T cache;
    typename CONFIG_T::accum_t mult[CONFIG_T::n_in*CONFIG_T::n_out];
    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];

    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases
    #pragma HLS function_instantiate variable=weights,biases

    if (CONFIG_T::io_type == io_parallel){
        // For parallel inputs:
        //   - completely partition arrays -- target fabric
        //   - if we have an unroll factor, limit number of multipliers
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

        // #pragma HLS ARRAY_PARTITION variable=weights complete // remove this line for now, it breaks compression sometimes
        #pragma HLS ARRAY_PARTITION variable=biases complete
        #pragma HLS ARRAY_PARTITION variable=mult complete
        #pragma HLS ARRAY_PARTITION variable=acc complete

        int multiplier_limit  = ceil(float(CONFIG_T::n_in*CONFIG_T::n_out) / float(CONFIG_T::reuse_factor)) - floor(float(CONFIG_T::n_zeros) / float(CONFIG_T::reuse_factor));
        #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation

    } else if (CONFIG_T::io_type == io_serial){
        #pragma HLS ARRAY_RESHAPE variable=weights complete dim=1
        #pragma HLS ARRAY_PARTITION variable=mult complete dim=1
        #pragma HLS ARRAY_PARTITION variable=acc complete dim=1
        #pragma HLS DATAFLOW
        #pragma HLS STREAM variable=mult depth=1
        #pragma HLS STREAM variable=acc depth=1
    }
    
    // Do the matrix-multiply
    Product1: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
        cache = data[ii];
        Product2: for(int jj = 0; jj < CONFIG_T::n_out; jj++) {
            if (CONFIG_T::io_type == io_serial) {
                int multiplier_limit  = ceil(float(CONFIG_T::n_out) / float(CONFIG_T::reuse_factor));
                #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
            }
 	    int index = ii*CONFIG_T::n_out+jj;
	    mult[index] = cache * weights[index];
        }
    }

    // Initialize accumulator with input biases
    ResetAccum: for(int iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS UNROLL
        }
        acc[iacc] = (typename CONFIG_T::accum_t) biases[iacc];
    }

    // Accumulate multiplication result
    Accum1: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
        Accum2: for(int jj = 0; jj < CONFIG_T::n_out; jj++) {
 	    int index = ii*CONFIG_T::n_out+jj;
	    acc[jj] += mult[index];
        }
    }

    // Cast to "res_t" type
    Result: for(int ires = 0; ires < CONFIG_T::n_out; ires++){
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS UNROLL
        }
        res[ires] = (res_T) (acc[ires]);
    }    
}

//Matrix operaions same as above, but with more options
template<class data_T, class res_T, unsigned int nin, unsigned int nout, typename CONFIG_T>
void matrixmult_Wb(
		data_T data[nin],
		res_T  tmpres [nout],
		typename CONFIG_T::weight_t     param_W[nin*nout],
		typename CONFIG_T::bias_t     param_b[nout]
		) 
{
  int multiplier_limit  = ceil(float(nin*nout) / float(CONFIG_T::reuse_factor)) - floor(float(CONFIG_T::n_zeros) / float(CONFIG_T::reuse_factor));
  #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
  data_T inputcache;
  res_T  acc[nout];
  typename CONFIG_T::weight_t inputmult[nin*nout];
  #pragma HLS function_instantiate variable=param_W,param_b
  #pragma HLS ARRAY_PARTITION variable=inputmult complete
  #pragma HLS ARRAY_PARTITION variable=param_b   complete
  #pragma HLS ARRAY_PARTITION variable=acc       complete

 Prod1: for(int ii = 0; ii < nin; ii++) {
    inputcache = data[ii];
  Prod2: for(int jj = 0; jj < nout; jj++) {
      int index=ii*nout+jj;
      inputmult[index] = inputcache * param_W[index];
    }
  }
  for(int ii = 0; ii < nout; ii++) acc[ii] = param_b[ii];
 Accum1: for(int ii = 0; ii < nin; ii++) {
  Accum2: for(int jj = 0; jj < nout; jj++) {
      int index = ii*nout + jj;
      acc[jj] += inputmult[index];
    }
  }
 Res: for(int ii = 0; ii < nout; ii++) {
    tmpres[ii] = acc[ii];
  }
}

//Matrix operaions
template<class data_T, class res_T, unsigned int nin, unsigned int nout, typename CONFIG_T>
void matrixmult_W(
		data_T data[nin],
		res_T  tmpres [nout],
		typename CONFIG_T::weight_t     param_W[nin*nout]
		) 
{
  int multiplier_limit  = ceil(float(nin*nout) / float(CONFIG_T::reuse_factor)) - floor(float(CONFIG_T::n_zeros) / float(CONFIG_T::reuse_factor));
  #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
  data_T inputcache;
  res_T  acc[nout];
  typename CONFIG_T::weight_t inputmult[nin*nout];
  #pragma HLS function_instantiate variable=param_W
  #pragma HLS ARRAY_PARTITION variable=inputmult complete
  #pragma HLS ARRAY_PARTITION variable=acc       complete
  #pragma HLS PIPELINE
  
 Prod1: for(int ii = 0; ii < nin; ii++) {
  #pragma HLS UNROLL
    inputcache = data[ii];
  Prod2: for(int jj = 0; jj < nout; jj++) {
  #pragma HLS UNROLL
      int index=ii*nout+jj;
      inputmult[index] = inputcache * param_W[index];
    }
  }
  for(int ii = 0; ii < nout; ii++) {
  #pragma HLS UNROLL
   acc[ii] = 0;
  }
 Accum1: for(int ii = 0; ii < nin; ii++) {
  #pragma HLS UNROLL
  Accum2: for(int jj = 0; jj < nout; jj++) {
  #pragma HLS UNROLL
      int index = ii*nout + jj;
      acc[jj] += inputmult[index];
    }
  }
 Res: for(int ii = 0; ii < nout; ii++) {
    #pragma HLS UNROLL
    tmpres[ii] = acc[ii];
  }
}
 template<class data_T, class res_T,unsigned int nin, unsigned int nout, typename CONFIG_T>
void matrixmult_Wb_2D(
		data_T data[nin],
		res_T  tmpres [nout],
		typename CONFIG_T::weight_t     param_W[nout][nin],
		typename CONFIG_T::bias_t     param_b[nout]
		) 
{
  // Operation: U*input
  data_T inputcache;
  typename CONFIG_T::weight_t inputmult[nin][nout];
  typename CONFIG_T::weight_t inputacc [nout];

 Prod1: for(int ii = 0; ii < nin; ii++) {
    inputcache = data[ii];
  Prod2: for(int jj = 0; jj < nout; jj++) {
      inputmult[ii][jj] = inputcache * param_W[jj][ii];
    }
  }
 Accum1: for(int ii = 0; ii < nin; ii++) {
  Accum2: for(int jj = 0; jj < nout; jj++) {
      tmpres[jj] += inputmult[ii][jj] + param_b[jj];
    }
  }
}
 template<class data_T, class res_T,unsigned int nin, unsigned int nout,typename CONFIG_T>
void matrixmult_W_2D(
		data_T tmpdata[CONFIG_T::n_state],
		res_T  tmpres [nout],
		typename CONFIG_T::weight_t     param_W[nout][CONFIG_T::n_state]
		) 
{
  // Operation: U*input
  data_T inputcache;
  typename CONFIG_T::weight_t inputmult[CONFIG_T::n_state][nout];
  typename CONFIG_T::weight_t inputacc [nout];

 Prod1: for(int ii = 0; ii < CONFIG_T::n_state; ii++) {
    inputcache = tmpdata[ii];
  Prod2: for(int jj = 0; jj < nout; jj++) {
      inputmult[ii][jj] = inputcache * param_W[jj][ii];
    }
  }
 Accum1: for(int ii = 0; ii < CONFIG_T::n_state; ii++) {
  Accum2: for(int jj = 0; jj < nout; jj++) {
      tmpres[jj] += inputmult[ii][jj];
    }
  }
}

}

#endif
