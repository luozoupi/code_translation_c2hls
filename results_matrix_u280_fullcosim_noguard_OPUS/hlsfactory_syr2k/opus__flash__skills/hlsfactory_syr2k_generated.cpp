#include "syr2k.h"


void kernel_syr2k( 
		  double alpha,
		  double beta,
		  double C[ N + 0][N + 0],
		  double A[ N + 0][M + 0],
		  double B[ N + 0][M + 0])
{
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=alpha bundle=control
#pragma HLS INTERFACE s_axilite port=beta  bundle=control
#pragma HLS INTERFACE s_axilite port=C     bundle=control
#pragma HLS INTERFACE s_axilite port=A     bundle=control
#pragma HLS INTERFACE s_axilite port=B     bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;
    const int m = M;

  int i, j, k;

  // Local staging buffers for reuse / parallel access
  double a_row[M];
  double b_row[M];
#pragma HLS ARRAY_PARTITION variable=a_row cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=b_row cyclic factor=8 dim=1

  double c_row[N];
#pragma HLS ARRAY_PARTITION variable=c_row cyclic factor=8 dim=1

  for (i = 0; i < n; i++) {

    // Stage A[i][*] and B[i][*] for reuse across the k/j loops
  load_ai:
    for (k = 0; k < m; k++) {
#pragma HLS PIPELINE II=1
      a_row[k] = A[i][k];
      b_row[k] = B[i][k];
    }

    // Stage current C row (only [0..i] valid, but load full for simplicity)
  load_c:
    for (j = 0; j <= i; j++) {
#pragma HLS PIPELINE II=1
      c_row[j] = C[i][j] * beta;
    }

  k_loop:
    for (k = 0; k < m; k++) {
      double bi = b_row[k];
      double ai = a_row[k];
    j_loop:
      for (j = 0; j <= i; j++) {
#pragma HLS PIPELINE II=1
        c_row[j] += A[j][k]*alpha*bi + B[j][k]*alpha*ai;
      }
    }

    // Write back the computed C row
  store_c:
    for (j = 0; j <= i; j++) {
#pragma HLS PIPELINE II=1
      C[i][j] = c_row[j];
    }
  }

}