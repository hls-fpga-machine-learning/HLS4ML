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

#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#include <complex>
#include "ap_int.h"
#include "ap_fixed.h"

#include "parameters.h"


// Prototype of top level function for C-synthesis
void myproject(
      input_t data[N_LOOP][N_INPUTS],
      result_t res[N_OUTPUTS],
      unsigned short &const_size_in,
      unsigned short &const_size_out);

void lstm_matrixmult_1 ( 
          input_t  data              [N_INPUTS],
          input_t  data_recurr       [N_STATE_1],
          layer1_t logits1     [N_STATE_1*4],
          layer1_t logitsnob1  [N_STATE_1*4],
          weight_default_t W1   [N_INPUTS*N_STATE_1*4],
          weight_default_t Wr1  [N_STATE_1*N_STATE_1*4],
          weight_default_t b1   [N_STATE_1*4]); 
void compute_layer2(layer1_t layer1_out[N_LAYER_1], result_t logits2[N_OUTPUTS]);

#endif

