#include "syrk.h"


void kernel_syrk( 
		 double alpha,
		 double beta,
		 double C[ N + 0][N + 0],
		 double A[ N + 0][M + 0])
{
  #pragma HLS top name=kernel_syrk

    const int n = N;
    const int m = M;

  int i, j, k;
  for (i = 0; i < n; i++) {
    for (j = 0; j <= i; j++)
      C[i][j] *= beta;
    for (k = 0; k < m; k++) {
      for (j = 0; j <= i; j++)
        C[i][j] += alpha * A[i][k] * A[j][k];
    }
  }

}