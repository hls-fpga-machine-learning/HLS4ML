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

#ifndef NNET_LARGELAYER_H_
#define NNET_LARGELAYER_H_

#include "nnet_common.h"
#include "hls_stream.h"
#include <math.h>

namespace nnet {

struct large_layer_config
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
    static const bool  use_lowlatency=false;
    // partitioning arrays cyclically to go with roll factors?
};

#define DIV_ROUNDUP(n,d) ((n + d - 1) / d)

 template<class data_T, class res_T, typename CONFIG_T>
void compute_large_layer(
    data_T    data[CONFIG_T::n_in],
    res_T     res[CONFIG_T::n_out],
    typename CONFIG_T::weight_t  weights[CONFIG_T::n_in*CONFIG_T::n_out],
    typename CONFIG_T::bias_t    biases[CONFIG_T::n_out]) {
    int      cycle_factor = DIV_ROUNDUP(CONFIG_T::n_in*CONFIG_T::n_out, CONFIG_T::reuse_factor);
    typename CONFIG_T::weight_t mult[CONFIG_T::n_in*CONFIG_T::n_out];
    if(CONFIG_T::use_lowlatency) { 
      int multiplier_limit  = ceil(float(CONFIG_T::n_in*CONFIG_T::n_out) / float(CONFIG_T::reuse_factor)) - floor(float(CONFIG_T::n_zeros) / float(CONFIG_T::reuse_factor));
      #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
      #pragma HLS PIPELINE II=CONFIG_T::reuse_factor rewind
    } 
    if(CONFIG_T::store_weights_in_bram) { 
      #pragma HLS RESOURCE        variable=weights core=ROM_1P_BRAM
    }
    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
    #pragma HLS function_instantiate variable=weights,biases
    #pragma HLS ARRAY_PARTITION variable=biases  complete
    #pragma HLS ARRAY_PARTITION variable=acc     complete
    #pragma HLS ARRAY_RESHAPE   variable=mult    block factor=cycle_factor
    #pragma HLS ARRAY_RESHAPE   variable=weights block factor=cycle_factor
    ResetAccum: for(int iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
        #pragma HLS UNROLL
        acc[iacc] = (typename CONFIG_T::accum_t) biases[iacc];
    }
    int rufactor=CONFIG_T::reuse_factor;
    if(CONFIG_T::use_lowlatency) { 
      rufactor          = CONFIG_T::n_in;
      cycle_factor      = CONFIG_T::n_out;
    }
    data_T cache;
    Product1: for(int ii = 0; ii < rufactor; ii++) {
       #pragma HLS PIPELINE II=1 rewind 
       if(CONFIG_T::use_lowlatency) { 
        cache = data[ii];
       }
    Product0: for(int jj = 0; jj < cycle_factor; jj++) {
        #pragma HLS UNROLL
        int windex = ii+jj*rufactor;
	int index   = windex/CONFIG_T::n_out;
	if(CONFIG_T::use_lowlatency) { 
         mult[windex] = cache*weights[windex];
	} else { 
 	 int aindex  = windex/CONFIG_T::n_in;
  	 acc[aindex] += data[index]*weights[windex];
	}
      }
    }
    if(CONFIG_T::use_lowlatency) { 
     // Accumulate multiplication result
     Accum1: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
         #pragma HLS UNROLL
         Accum2: for(int jj = 0; jj < CONFIG_T::n_out; jj++) {
            #pragma HLS UNROLL
	    int index = ii*CONFIG_T::n_out+jj;
	    acc[jj] += mult[index];
         }
     }
    }
    Result: for(int ires = 0; ires < CONFIG_T::n_out; ires++){
        #pragma HLS UNROLL
        res[ires] = (res_T) (acc[ires]);
    }    
}

}

#endif
