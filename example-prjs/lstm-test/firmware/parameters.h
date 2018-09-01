#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <complex>
#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_layer.h"
#include "nnet_sublayer.h"
#include "nnet_conv.h"
#include "nnet_conv2d.h"
#include "nnet_recursive.h"
#include "nnet_activation.h"
#include "nnet_common.h"

//hls-fpga-machine-learning insert numbers
typedef ap_fixed<18,8> accum_default_t;
typedef ap_fixed<18,8> weight_default_t;
typedef ap_fixed<18,8> bias_default_t;
typedef ap_fixed<18,8> input_t;
typedef ap_fixed<18,8> result_t;
#define N_LOOP    20
#define N_INPUTS  6
#define N_LAYER_1 16
#define N_STATE_1 16
#define N_OUTPUTS 5

//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<18,8> layer1_t;

//hls-fpga-machine-learning insert layer-config
struct config1 : nnet::lstm_config {
        typedef accum_default_t accum_t;
        typedef weight_default_t weight_t;  // Matrix
        typedef bias_default_t   bias_t;  // Vector
        static const unsigned n_in  = N_INPUTS;
        static const unsigned n_out = N_LAYER_1;
        static const unsigned n_state = N_STATE_1;
        static const unsigned io_type = nnet::io_parallel;
        static const unsigned reuse_factor = 1;
        static const bool     store_weights_in_bram = false;        
        };
struct sigmoid_config1_lstm : nnet::activ_config {
        static const unsigned n_in = N_LAYER_1*3;
        static const unsigned table_size = 1024;
        static const unsigned io_type = nnet::io_parallel;
        static const unsigned activation_type = nnet::activ_sigmoid;
        };
struct relu_config1 : nnet::activ_config {
        static const unsigned n_in = N_LAYER_1;
        static const unsigned table_size = 1024;
        static const unsigned io_type = nnet::io_parallel;
        static const unsigned activation_type = nnet::activ_relu;
        };
struct config2_0 : nnet::sublayer_config {
        static const unsigned n_in = N_LAYER_1;
        static const unsigned n_out = N_OUTPUTS;
        static const unsigned n_part = 2;
        static const unsigned i_part = 0;
        static const unsigned n_sub_out = 3;
        static const unsigned i_sub_out = 0;
        static const unsigned io_type = nnet::io_parallel;
        static const unsigned reuse_factor = 1;
        static const unsigned n_zeros = 0;
        static const bool store_weights_in_bram = false;
        typedef accum_default_t accum_t;
        typedef bias_default_t bias_t;
        typedef weight_default_t weight_t;
        };
struct config2_1 : nnet::sublayer_config {
        static const unsigned n_in = N_LAYER_1;
        static const unsigned n_out = N_OUTPUTS;
        static const unsigned n_part = 2;
        static const unsigned i_part = 1;
        static const unsigned n_sub_out = 2;
        static const unsigned i_sub_out = 3;
        static const unsigned io_type = nnet::io_parallel;
        static const unsigned reuse_factor = 1;
        static const unsigned n_zeros = 0;
        static const bool store_weights_in_bram = false;
        typedef accum_default_t accum_t;
        typedef bias_default_t bias_t;
        typedef weight_default_t weight_t;
        };
struct softmax_config2 : nnet::activ_config {
        static const unsigned n_in = N_OUTPUTS;
        static const unsigned table_size = 1024;
        static const unsigned io_type = nnet::io_parallel;
        static const unsigned activation_type = nnet::activ_softmax;
        };
#endif 
