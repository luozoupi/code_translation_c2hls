#include "kernel.h"

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=256
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=256
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=256
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
for (int i = 0; i < I; i++)
    for (int j = 0; j < J; j++) {
      //C[i][j] = (i == 3 && j == 5) ? 1.0f : 0;
      for (int k = 0; k < K; k++) {        
        C[i][j] = (i == 3 && j == 5) ? 1.0f : C[i][j] + A[i][k] * B[j][k];
      }
    }
}
