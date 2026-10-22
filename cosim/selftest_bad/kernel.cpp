#include "kernel.h"

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
for (int i = 0; i < I; i++)
    for (int j = 0; j < J; j++) {
      //C[i][j] = (i == 3 && j == 5) ? 1.0f : 0;
      for (int k = 0; k < K; k++) {        
        C[i][j] = (i == 3 && j == 5) ? 1.0f : C[i][j] + A[i][k] * B[j][k];
      }
    }
}
