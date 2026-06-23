#include "atax.h"


void kernel_atax( 
		 double A[ M + 0][N + 0],
		 double x[ N + 0],
		 double y[ N + 0],
		 double tmp[ M + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=y offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=tmp offset=slave bundle=gmem3
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=x bundle=control
#pragma HLS INTERFACE s_axilite port=y bundle=control
#pragma HLS INTERFACE s_axilite port=tmp bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int m = M;
    const int n = N;

  int i, j;

  // Local buffers to enable reuse and parallel access
  double x_local[N];
  double y_local[N];
#pragma HLS ARRAY_PARTITION variable=x_local cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=y_local cyclic factor=8 dim=1

  // Stage x into local memory and initialize y_local
LOAD_X:
  for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
    x_local[j] = x[j];
  }

INIT_Y:
  for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
    y_local[i] = 0;
  }

OUTER:
  for (i = 0; i < m; i++)
    {
      double A_row[N];
#pragma HLS ARRAY_PARTITION variable=A_row cyclic factor=8 dim=1

      // Stage the current row of A into local memory for reuse
    LOAD_ROW:
      for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
        A_row[j] = A[i][j];
      }

      double t = 0.0;
    DOT:
      for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
        t = t + A_row[j] * x_local[j];
      }
      tmp[i] = t;

    ACCUM_Y:
      for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
        y_local[j] = y_local[j] + A_row[j] * t;
      }
    }

  // Write back y
STORE_Y:
  for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
    y[j] = y_local[j];
  }

}