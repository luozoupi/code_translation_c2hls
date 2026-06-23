#include "cholesky.h"


void kernel_cholesky(
		     double A[ N + 0][N + 0])
{


    const int n = N;
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  int i, j, k;

  // Stage the matrix into a local buffer to enable reuse and avoid
  // repeated global-memory accesses across the triangular update.
  static double L[N][N];
#pragma HLS ARRAY_PARTITION variable=L cyclic factor=8 dim=2

  // Copy input from global memory into local buffer.
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
      L[i][j] = A[i][j];
    }
  }

  for (i = 0; i < n; i++) {

     for (j = 0; j < i; j++) {
        double acc = L[i][j];
        for (k = 0; k < j; k++) {
#pragma HLS PIPELINE II=1
           acc -= L[i][k] * L[j][k];
        }
        L[i][j] = acc / L[j][j];
     }

     double diag = L[i][i];
     for (k = 0; k < i; k++) {
#pragma HLS PIPELINE II=1
        diag -= L[i][k] * L[i][k];
     }
     L[i][i] = sqrt(diag);
  }

  // Write the result back to global memory.
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
      A[i][j] = L[i][j];
    }
  }

}