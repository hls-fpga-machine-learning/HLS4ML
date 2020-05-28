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
#include <algorithm>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "firmware/myproject.h"


int main(int argc, char **argv)
{
  //load input data from text file
  std::ifstream fin("tb_data/tb_input_features.dat");
  std::ifstream fri("tb_data/tb_adjacency_incoming.dat");
  std::ifstream fro("tb_data/tb_adjacency_outgoing.dat");
  //load predictions from text file
  std::ifstream fpr("tb_data/tb_output_predictions.dat");

#ifdef RTL_SIM
  std::string RESULTS_LOG = "tb_data/rtl_cosim_results.log";
#else
  std::string RESULTS_LOG = "tb_data/csim_results.log";
#endif
  std::ofstream fout(RESULTS_LOG);

  std::string iline;
  std::string riline;
  std::string roline;
  std::string pline;
  int e = 0;

  while ( std::getline(fin,iline) && std::getline(fri,riline) && std::getline(fro,roline) && std::getline (fpr,pline) ) {
    std::cout << "Processing input " << e << std::endl;
    
    char* cstr=const_cast<char*>(iline.c_str());
    char* current;
    std::vector<float> in;
    current=strtok(cstr," ");
    while(current!=NULL) {
      in.push_back(atof(current));
      current=strtok(NULL," ");
    }
    
    cstr=const_cast<char*>(riline.c_str());
    std::vector<uint> ri;
    current=strtok(cstr," ");
    while(current!=NULL) {
      ri.push_back(atof(current));
      current=strtok(NULL," ");
    }
    
    cstr=const_cast<char*>(roline.c_str());
    std::vector<uint> ro;
    current=strtok(cstr," ");
    while(current!=NULL) {
      ro.push_back(atof(current));
      current=strtok(NULL," ");
    }
    
    cstr=const_cast<char*>(pline.c_str());
    std::vector<float> pr;
    current=strtok(cstr," ");
    while(current!=NULL) {
      pr.push_back(atof(current));
      current=strtok(NULL," ");
    }
    
    //hls-fpga-machine-learning insert data
    input_t X_str[N_NODES][N_FEATURES];
    for(int i = 0; i < N_NODES; i++) {
      for(int j = 0; j < N_FEATURES; j++) {
	X_str[i][j] = in[j+N_FEATURES*i];
      }
    }

    ap_uint<1> Ri_str[N_NODES][N_EDGES];
    for(int i = 0; i < N_NODES; i++) {
      for(int j = 0; j < N_EDGES; j++) {
	Ri_str[i][j] = ri[j+N_EDGES*i];
      }
    }

    ap_uint<1> Ro_str[N_NODES][N_EDGES];
    for(int i = 0; i < N_NODES; i++) {
      for(int j = 0; j < N_EDGES; j++) {
	Ro_str[i][j] = ro[j+N_EDGES*i];
      }
    }

    result_t e_str[N_EDGES][1];
    for(int i=0; i<N_EDGES; i++){
      e_str[i][0]=0;
    }

    unsigned short size_in, size_out;
    myproject(X_str, Ri_str, Ro_str, e_str, size_in, size_out);
    
    //hls-fpga-machine-learning insert tb-output
    for(int i = 0; i < N_EDGES; i++) {
      fout << e_str[i][0] << " ";
    }
    fout << std::endl;
    
    std::cout << "Predictions" << std::endl;
    //hls-fpga-machine-learning insert predictions
    for(int i = 0; i < N_EDGES; i++) {
      std::cout << pr[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Quantized predictions" << std::endl;
    //hls-fpga-machine-learning insert quantized
    for(int i = 0; i < N_EDGES; i++) {
      std::cout << e_str[i][0] << " ";
      }
      std::cout << std::endl;
  }
  fin.close();
  fri.close();
  fro.close();
  fpr.close();  
  fout.close();
  std::cout << "INFO: Saved inference results to file: " << RESULTS_LOG << std::endl;
  
  return 0;
}


