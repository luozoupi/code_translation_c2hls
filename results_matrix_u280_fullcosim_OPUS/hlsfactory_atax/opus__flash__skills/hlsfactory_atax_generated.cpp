#include "atax.h"


void kernel_atax( 
		 double A[ M + 0][N + 0],
		 double x[ N + 0],
		 double y[ N + 0],
		 double tmp[ M + 0])
{
#pragma HLS INLINE off

    const int m = M;
    const int n = N;

  int i, j;

  init_y: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
    y[i] = 0;
  }

  outer: for (i = 0; i < m; i++)
    {
      double tmp_local = 0.0;

      inner1: for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
	tmp_local = tmp_local + A[i][j] * x[j];
      }
      tmp[i] = tmp_local;

      inner2: for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
	y[j] = y[j] + A[i][j] * tmp_local;
      }
    }

}

extern "C" {
void workload(
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

    kernel_atax(A, x, y, tmp);
}
}