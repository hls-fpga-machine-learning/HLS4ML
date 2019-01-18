#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <complex>
#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_large_layer.h"
#include "nnet_conv.h"
#include "nnet_conv2d.h"
#include "nnet_activation.h"
#include "nnet_common.h"
#include "nnet_batchnorm.h"
#include "nnet_pooling.h"

//hls-fpga-machine-learning insert numbers
typedef ap_fixed<16,6> accum_default_t;
typedef ap_fixed<16,6> weight_default_t;
typedef ap_fixed<16,6> bias_default_t;
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<16,6> result_t;
#define N_INPUTS 110
#define N_LAYER_1 150
#define N_LAYER_2 50
#define N_OUTPUTS 10

//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> layer1_t;
typedef ap_fixed<16,6> layer2_t;

//hls-fpga-machine-learning insert layer-config
struct config1 : nnet::large_layer_config {
        static const unsigned n_in = N_INPUTS;
        static const unsigned n_out = N_LAYER_1;
        static const unsigned io_type = nnet::io_parallel;
        static const unsigned reuse_factor = 110; //max reuse without modified setup
        static const unsigned n_zeros = 14850;
        static const bool store_weights_in_bram = true;
        typedef accum_default_t accum_t;
        typedef bias_default_t bias_t;
        typedef weight_default_t weight_t;
        };
struct linear_config1 : nnet::activ_config {
        static const unsigned n_in = N_LAYER_1;
        static const unsigned table_size = 1024;
        static const unsigned io_type = nnet::io_parallel;
        };
struct config2 : nnet::large_layer_config {
        static const unsigned n_in = N_LAYER_1;
        static const unsigned n_out = N_LAYER_2;
        static const unsigned io_type = nnet::io_parallel;
        static const unsigned reuse_factor = 150; //max resue without modified setup
        static const unsigned n_zeros = 6750;
        static const bool store_weights_in_bram = true;
        typedef accum_default_t accum_t;
        typedef bias_default_t bias_t;
        typedef weight_default_t weight_t;
        };
struct linear_config2 : nnet::activ_config {
        static const unsigned n_in = N_LAYER_2;
        static const unsigned table_size = 1024;
        static const unsigned io_type = nnet::io_parallel;
        };
struct config3 : nnet::large_layer_config {
        static const unsigned n_in = N_LAYER_2;
        static const unsigned n_out = N_OUTPUTS;
        static const unsigned io_type = nnet::io_parallel;
        static const unsigned reuse_factor = 50; //max reuse without modified setup
        static const unsigned n_zeros = 450;
        static const bool store_weights_in_bram = true;
        typedef accum_default_t accum_t;
        typedef bias_default_t bias_t;
        typedef weight_default_t weight_t;
        };
struct linear_config3 : nnet::activ_config {
        static const unsigned n_in = N_OUTPUTS;
        static const unsigned table_size = 1024;
        static const unsigned io_type = nnet::io_parallel;
        };

#endif 
