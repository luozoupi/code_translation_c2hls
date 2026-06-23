#include "syr2k.h"

extern "C" {
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

  // Cache full A and B rows locally to avoid repeated AXI reads and
  // memory-port conflicts inside the inner loops.
  static double A_buf[N][M];
  static double B_buf[N][M];
#pragma HLS ARRAY_PARTITION variable=A_buf cyclic factor=2 dim=2
#pragma HLS ARRAY_PARTITION variable=B_buf cyclic factor=2 dim=2

  for (i = 0; i < n; i++) {
    for (k = 0; k < m; k++) {
#pragma HLS PIPELINE II=1
      A_buf[i][k] = A[i][k];
      B_buf[i][k] = B[i][k];
    }
  }

  for (i = 0; i < n; i++) {
    // Local accumulation row, initialized with beta-scaled C.
    double C_row[N];
#pragma HLS ARRAY_PARTITION variable=C_row cyclic factor=4 dim=1

    for (j = 0; j <= i; j++) {
#pragma HLS PIPELINE II=1
      C_row[j] = C[i][j] * beta;
    }

    for (k = 0; k < m; k++) {
      double a_ik = A_buf[i][k];
      double b_ik = B_buf[i][k];
      for (j = 0; j <= i; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=C_row inter false
	C_row[j] += A_buf[j][k]*alpha*b_ik + B_buf[j][k]*alpha*a_ik;
      }
    }

    // Write back the computed row.
    for (j = 0; j <= i; j++) {
#pragma HLS PIPELINE II=1
      C[i][j] = C_row[j];
    }
  }

}
}