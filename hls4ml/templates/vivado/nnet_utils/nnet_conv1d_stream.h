#ifndef NNET_CONV1D_STREAM_H_
#define NNET_CONV1D_STREAM_H_

#include "nnet_common.h"
#include "nnet_conv_stream.h"
#include "hls_stream.h"

namespace nnet {

template<class data_T, typename CONFIG_T>
void compute_scaled_indices_1d(
    const unsigned w_idx,
    ap_uint<CONFIG_T::filt_width> *pixel_idx
) {
    unsigned wp_idx = w_idx * (data_T::size / CONFIG_T::n_chan);

    ComputeIndex: for (unsigned p = 0; p < data_T::size / CONFIG_T::n_chan; p++) {
        #pragma HLS UNROLL

        unsigned sw_idx = scale_index<CONFIG_T::filt_width, CONFIG_T::stride_width, CONFIG_T::in_width>(wp_idx + p);
        pixel_idx[p] = CONFIG_T::pixels[sw_idx];
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void conv_1d_cl(
    hls::stream<data_T> &data,
    hls::stream<res_T>  &res,
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt])
{
    assert(CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);
    assert(CONFIG_T::stride_width <= CONFIG_T::filt_width);

    hls::stream<typename data_T::value_type> data_window[CONFIG_T::filt_width * CONFIG_T::n_chan];
    const int win_depth = CONFIG_T::out_width;
    for (unsigned i_out = 0; i_out < CONFIG_T::filt_width * CONFIG_T::n_chan; i_out++) {
        #pragma HLS STREAM variable=data_window[i_out] depth=win_depth
    }

    #pragma HLS ARRAY_PARTITION variable=CONFIG_T::pixels complete

    res_T res_pack;
    #pragma HLS DATA_PACK variable=res_pack
    unsigned outputs_ready = 0;

    ap_uint<CONFIG_T::filt_width> pixel_idx[data_T::size / CONFIG_T::n_chan];
    #pragma HLS ARRAY_PARTITION variable=pixel_idx complete

    ReadInputWidth: for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width / (data_T::size / CONFIG_T::n_chan); i_iw++) {
        #pragma HLS LOOP_FLATTEN
        if (CONFIG_T::strategy == nnet::latency && data_T::size / CONFIG_T::n_chan == 1) {
            #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
        }
        compute_scaled_indices_1d<data_T, CONFIG_T>(i_iw, pixel_idx);
        compute_output<data_T, res_T, CONFIG_T>(data.read(), data_window, res, res_pack, outputs_ready, weights, biases, pixel_idx);
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void single_output_conv_1d_cl(
    hls::stream<data_T> &data,
    hls::stream<res_T>  &res,
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt])
{
    assert(CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);
    assert(CONFIG_T::filt_width == CONFIG_T::in_width);
    assert(CONFIG_T::out_width == 1);

    typename data_T::value_type data_window[CONFIG_T::filt_width * CONFIG_T::n_chan];
    #pragma HLS ARRAY_PARTITION variable=data_window complete
    typename res_T::value_type res_elem[CONFIG_T::n_filt];
    #pragma HLS ARRAY_PARTITION variable=res_elem complete
    res_T res_pack;
    #pragma HLS DATA_PACK variable=res_pack

    ReadInputImage: for (unsigned i = 0; i < CONFIG_T::in_width; i++) {
        #pragma HLS PIPELINE
        data_T in_elem = data.read();
        CopyDataChan: for (unsigned c = 0; c < CONFIG_T::n_chan; c++) {
            #pragma HLS UNROLL
            data_window[i * CONFIG_T::n_chan + c] = in_elem[c];
        }
    }

    #pragma HLS INLINE region
    if (CONFIG_T::strategy == nnet::latency) {
        dense_latency<typename data_T::value_type, typename res_T::value_type, typename CONFIG_T::mult_config>(data_window, res_elem, weights, biases);
    } else {
        dense_resource<typename data_T::value_type, typename res_T::value_type, typename CONFIG_T::mult_config>(data_window, res_elem, weights, biases);
    }

    CastLoop: for (unsigned jj = 0; jj < CONFIG_T::n_filt; jj++) {
        #pragma HLS UNROLL
        res_pack[jj] = res_elem[jj];
    }

    res.write(res_pack);
}

}
#endif
