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
#pragma HLS INTERFACE s_axilite port=beta bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;
    const int m = M;

  int i, j, k;

  // Local buffers to keep the read-modify-write accumulation in BRAM
  // instead of going through AXI for every C[i][j] update.
  static double C_row[N];
  static double A_row[M];   // A[i][*]
  static double B_row[M];   // B[i][*]
  static double A_loc[N][M];
  static double B_loc[N][M];
#pragma HLS ARRAY_PARTITION variable=A_row complete dim=1
#pragma HLS ARRAY_PARTITION variable=B_row complete dim=1

  // Stage A and B fully into local memory once (each element read once).
  for (i = 0; i < n; i++) {
    for (k = 0; k < m; k++) {
#pragma HLS PIPELINE II=1
      A_loc[i][k] = A[i][k];
      B_loc[i][k] = B[i][k];
    }
  }

  for (i = 0; i < n; i++) {

    // Load current C row and apply beta scaling locally.
    for (j = 0; j <= i; j++) {
#pragma HLS PIPELINE II=1
      C_row[j] = C[i][j] * beta;
    }

    // Cache A[i][*] and B[i][*] rows for reuse across the j loop.
    for (k = 0; k < m; k++) {
#pragma HLS PIPELINE II=1
      A_row[k] = A_loc[i][k];
      B_row[k] = B_loc[i][k];
    }

    // Main accumulation. The reduction over k into C_row[j] is preserved
    // in its original serial order (no unroll / no reassociation).
    for (k = 0; k < m; k++) {
      for (j = 0; j <= i; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=C_row inter false
	C_row[j] += A_loc[j][k]*alpha*B_row[k] + B_loc[j][k]*alpha*A_row[k];
      }
    }

    // Write the finished C row back to global memory once.
    for (j = 0; j <= i; j++) {
#pragma HLS PIPELINE II=1
      C[i][j] = C_row[j];
    }
  }

}