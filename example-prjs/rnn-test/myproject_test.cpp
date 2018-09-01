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
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "firmware/parameters.h"
#include "firmware/myproject.h"
#include "nnet_helpers.h"

int main(int argc, char **argv)
{
    
    //hls-fpga-machine-learning insert data
    //input_t  data_str[Y_INPUTS][N_CHAN];
    //for(int i=0; i<Y_INPUTS; i++){
    //    for(int j=0; j<N_CHAN; j++){
    //	    data_str[i][j]=1;
    //	}
    //}
  mytype data[N_LOOP][N_INPUTS] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};//, 11, 12, 13, 14, 15 ,16 ,17, 18, 19, 20};
  mytype res[N_LOOP][N_OUTPUTS];

    unsigned short size_in, size_out;
    myproject(data, res, size_in, size_out);

    // result_t res_expected[N_OUTPUTS] = {0.528625502721, 0.352447456382, 0.11757160967, 0.000254924257721, 8.45225898784e-05, 0.00101598437936};
    
    // for(int i=0; i<N_OUTPUTS; i++){
    //     std::cout << res_str[i] << " (expected " << res_expected[i] << ", " << 100.0*((float)res_str[i]-(float)res_expected[i])/(float)res_expected[i] << " percent difference)" << std::endl;
    // }
    //std::cout << std::endl;
    
    return 0;
}
