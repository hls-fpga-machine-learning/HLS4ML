//Numpy array shape [3, 4]
//Min -1.164089441299
//Max 1.771318197250
//Number of zeros 0

#ifndef W1_H_
#define W1_H_

#ifndef __SYNTHESIS__
ap_fixed<16,6> w1[12];
#else
ap_fixed<16,6> w1[12] = {-0.213103, 1.579169, -0.123396, -1.164089, 0.020248, -0.090968, 0.934973, 1.771318, -0.012138, -0.004056, -0.033154, -0.066430};
#endif

#endif
