#ifndef DEFINES_H_
#define DEFINES_H_

#include <complex>
#include "ap_int.h"
#include "ap_fixed.h"

//hls-fpga-machine-learning insert numbers
#define REUSE 10
#define N_FEATURES 3
#define N_HIDDEN_FEATURES 4
//2x2 example:
//#define N_NODES 4
//#define N_EDGES 4
//3x3 example:
#define N_NODES 9
#define N_EDGES 18
//4x4 example:
//#define N_NODES 16
//#define N_EDGES 48
//5x5 example:
//#define N_NODES 25
//#define N_EDGES 100

//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<16,6> accum_default_t;
typedef ap_fixed<16,6> weight_default_t;
typedef ap_fixed<16,6> bias_default_t;
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<16,6> result_t;

#endif
