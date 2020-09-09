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

#include "myproject.h"

//hls4ml init engine
//hls4ml init nn stream
dnnl::stream engine_stream(eng); 
std::vector<dnnl::primitive> net; // neural network as a vector
std::vector<std::unordered_map<int, dnnl::memory>> net_args; // nn arguments
dnnl::memory input_data_memory;
dnnl::memory output_memory;

extern "C" {

    void compile_model() {

        //hls4ml insert layers

    }
    
    void myproject_float(float *input_data, float *output_data) {

        //hls4ml execute network
        write_to_dnnl_memory(input_data, input_data_memory);
        for (size_t i = 0; i < net.size(); ++i)
                net.at(i).execute(engine_stream, net_args.at(i));
        engine_stream.wait();

        //hls4ml read output data from memory
    }

}