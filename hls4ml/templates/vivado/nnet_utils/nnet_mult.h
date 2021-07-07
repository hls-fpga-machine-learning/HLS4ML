#ifndef NNET_MULT_H_
#define NNET_MULT_H_

#include "nnet_common.h"
#include "nnet_helpers.h"
#include "hls_stream.h"
#include <math.h>

namespace nnet {

namespace product{

/* ---
 * 5 different methods to perform the product of input and weight, depending on the
 * types of each. 
 * --- */

template<class x_T, class w_T, class y_T>
class Product{
    public:
    static y_T product(x_T a, w_T w){
        // 'Normal' product
        #pragma HLS INLINE
        return a * w;
    }
    static void limit(unsigned multiplier_limit) {} // Nothing to do here
};

template<class x_T, class w_T, class y_T>
class both_binary : public Product<x_T, w_T, y_T>{
    public:
    static y_T product(x_T a, w_T w){
        // specialisation for 1-bit weights and incoming data
        #pragma HLS INLINE
        return a == w;
    }
};

template<class x_T, class w_T, class y_T>
class weight_binary : public Product<x_T, w_T, y_T>{
    public:
    static y_T product(x_T a, w_T w){
        // Specialisation for 1-bit weights, arbitrary data
        #pragma HLS INLINE
        return w == 0 ? (x_T) -a : a;
    }
};

template<class x_T, class w_T, class y_T>
class data_binary : public Product<x_T, w_T, y_T>{
    public:
    static y_T product(x_T a, w_T w){
        // Specialisation for 1-bit data, arbitrary weight
        #pragma HLS INLINE
        return a == 0 ? (w_T) -w : w;
    }
};

template<class x_T, class w_T, class y_T>
class weight_ternary : public Product<x_T, w_T, y_T>{
    public:
    static y_T product(x_T a, w_T w){
        // Specialisation for 2-bit weights, arbitrary data
        #pragma HLS INLINE
        if (w == 0) return (x_T) 0;
        else if(w == -1) return (x_T) -a;
        else return (x_T) a; // if(w == 1)
    }
};

template<class x_T, class w_T, class y_T>
class mult : public Product<x_T, w_T, y_T>{
    public:
    static y_T product(x_T a, w_T w){
        // 'Normal' product
        #pragma HLS INLINE
        return a * w;
    }
    static void limit(unsigned multiplier_limit){
        #pragma HLS INLINE
        #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
    }
};

template<class x_T, class w_T, class y_T>
class weight_exponential : public Product<x_T, w_T, y_T>{
    public:
    static y_T product(x_T a, w_T w){
        // Shift product for exponential weights
        #pragma HLS INLINE
        // shift by the exponent. Negative weights shift right
        y_T y = a << w.weight;
        // negate or not depending on weight sign
        return w.sign == 1 ? (y_T) y : (y_T) -y;
    }
};

} // namespace product_type

template<class data_T, class res_T, typename CONFIG_T>
inline typename std::enable_if<std::is_same<data_T, ap_uint<1>>::value
        && std::is_same<typename CONFIG_T::weight_t, ap_uint<1>>::value, ap_int<nnet::ceillog2(CONFIG_T::n_in) + 2>>::type
cast(typename CONFIG_T::accum_t x){
  return (ap_int<nnet::ceillog2(CONFIG_T::n_in) + 2>) (x - CONFIG_T::n_in / 2) * 2;
}

template<class data_T, class res_T, typename CONFIG_T>
inline typename std::enable_if<std::is_same<data_T, ap_uint<1>>::value
        && ! std::is_same<typename CONFIG_T::weight_t, ap_uint<1>>::value, res_T>::type
cast(typename CONFIG_T::accum_t x){
  return (res_T) x;
}

template<class data_T, class res_T, typename CONFIG_T>
inline typename std::enable_if<(! std::is_same<data_T, ap_uint<1>>::value), res_T>::type
cast(typename CONFIG_T::accum_t x){
  return (res_T) x;
}

}

#endif
