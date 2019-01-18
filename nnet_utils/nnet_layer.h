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
    static const bool use_lowlatency=true;
    // partitioning arrays cyclically to go with roll factors?
};

#define DIV_ROUNDUP(n,d) ((n + d - 1) / d)

 template<class data_T, class res_T, typename CONFIG_T>
void compute_layer(
    data_T    data[CONFIG_T::n_in],
    res_T     res[CONFIG_T::n_out],
    typename CONFIG_T::weight_t  weights[CONFIG_T::n_in*CONFIG_T::n_out],
    typename CONFIG_T::bias_t    biases[CONFIG_T::n_out])
{
    static const int mult_factor = DIV_ROUNDUP(CONFIG_T::n_in*CONFIG_T::n_out, CONFIG_T::reuse_factor);
    static const int nmult       = DIV_ROUNDUP(mult_factor, CONFIG_T::n_out);
    static const int nmults      = DIV_ROUNDUP(mult_factor, CONFIG_T::n_out)*CONFIG_T::n_out;
    int mult_factor_loop         = mult_factor;
    typename CONFIG_T::accum_t mult[nmults];
    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases
    #pragma HLS function_instantiate variable=weights,biases
    if (CONFIG_T::io_type == io_parallel){
      // For parallel inputs:
      //   - completely partition arrays -- target fabric
      //   - if we have an unroll factor, limit number of multipliers
      if(CONFIG_T::store_weights_in_bram) { 
       #pragma HLS RESOURCE        variable=weights core=ROM_1P_BRAM
      }
      #pragma HLS ARRAY_PARTITION variable=biases complete
      #pragma HLS ARRAY_PARTITION variable=acc    complete
      #pragma HLS ARRAY_RESHAPE   variable=mult   complete
      #pragma HLS ARRAY_RESHAPE   variable=weights block factor=mult_factor
      if(CONFIG_T::use_lowlatency) { 
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
	int multiplier_limit  = ceil(float(CONFIG_T::n_in*CONFIG_T::n_out) / float(CONFIG_T::reuse_factor)) - floor(float(CONFIG_T::n_zeros) / float(CONFIG_T::reuse_factor));
        #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
       }
    } else if (CONFIG_T::io_type == io_serial){
        // Only reduce cycle_factor if n_out is evenly divisible by reuse_factor
        // Otherwise, HLS wont be happy
        int cycle_factor = CONFIG_T::n_out;
        float reused_cycle = CONFIG_T::n_out / CONFIG_T::reuse_factor;
        if (reused_cycle == ceil(reused_cycle)){
            // Dont use "ceil" here; as of 2018.2, HLS crashes mysteriously
            cycle_factor = cycle_factor / CONFIG_T::reuse_factor;
        }
        #pragma HLS ARRAY_PARTITION variable=weights cyclic factor=cycle_factor
        #pragma HLS ARRAY_PARTITION variable=mult cyclic factor=cycle_factor
        #pragma HLS ARRAY_PARTITION variable=acc complete
        #pragma HLS DATAFLOW
        #pragma HLS STREAM variable=mult depth=1
        #pragma HLS STREAM variable=acc depth=1
        if (CONFIG_T::store_weights_in_bram){
            #pragma HLS RESOURCE variable=weights core=ROM_2P_BRAM
        }
    }
    ResetAccum: for(int iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
        #pragma HLS UNROLL
      acc[iacc] = (typename CONFIG_T::accum_t) biases[iacc];
    }
    ResetMult: for(int imult = 0; imult < nmults; imult++) {
        #pragma HLS UNROLL
      mult[imult] = 0;
    }
    int rufactor=CONFIG_T::reuse_factor;
    if(CONFIG_T::use_lowlatency) { 
      rufactor          = CONFIG_T::n_in;
      mult_factor_loop  = CONFIG_T::n_out;
    }
    data_T cache;
    Product1: for(int ii = 0; ii < rufactor; ii++) {
       #pragma HLS PIPELINE II=1 rewind 
       if(CONFIG_T::use_lowlatency) { 
	 cache = data[ii];
       }
    Product0: for(int jj = 0; jj < mult_factor_loop; jj++) {
	 if (CONFIG_T::io_type == io_serial) {
	   int multiplier_limit  = ceil(float(CONFIG_T::n_out) / float(CONFIG_T::reuse_factor));
           #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
	 } else { //PH Check Me 
           #pragma HLS UNROLL 
	 }
	 int windex = ii+jj*rufactor;
	 int index   = windex/CONFIG_T::n_out;
	 if(CONFIG_T::use_lowlatency) { 
	   mult[windex] = cache*weights[windex];
	 } else { 
           int aindex  = (nmult*windex)/CONFIG_T::n_in;
	   mult[aindex] += data[index]*weights[windex];
	 }
       }
    }
    // Accumulate multiplication result
    Accum1: for(int ii = 0; ii < CONFIG_T::n_out; ii++) {
     if (CONFIG_T::io_type == io_serial){
        #pragma HLS PIPELINE
      } else { 
        #pragma HLS UNROLL
      }
      Accum2: for(int jj = 0; jj < nmult; jj++) {
       #pragma HLS UNROLL
       int index = ii*nmult+jj;
       acc[ii]  += mult[index];
      }
    }
    // Cast to "res_t" type
    Result: for(int ires = 0; ires < CONFIG_T::n_out; ires++){
       #pragma HLS UNROLL 
       if (CONFIG_T::io_type == io_serial){
            #pragma HLS UNROLL
        }
        res[ires] = (res_T) (acc[ires]);
    }    
}

}

#endif
