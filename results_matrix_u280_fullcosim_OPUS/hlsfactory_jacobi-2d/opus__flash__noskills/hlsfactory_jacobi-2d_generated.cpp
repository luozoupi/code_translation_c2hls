#include "jacobi-2d.h"

extern "C" {
void kernel_jacobi_2d(

			    double A[ N + 0][N + 0],
			    double B[ N + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;
    const int tsteps = TSTEPS;

  int t, i, j;

  // Local on-chip buffers to avoid AXI port contention in inner loops.
  static double la[N][N];
  static double lb[N][N];
#pragma HLS ARRAY_PARTITION variable=la cyclic factor=2 dim=2
#pragma HLS ARRAY_PARTITION variable=lb cyclic factor=2 dim=2

  // Copy A from global memory into local buffer.
  copy_in_i: for (i = 0; i < n; i++)
    copy_in_j: for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
      la[i][j] = A[i][j];
      lb[i][j] = B[i][j];
    }

  for (t = 0; t < tsteps; t++)
    {
      for (i = 1; i < n - 1; i++)
	for (j = 1; j < n - 1; j++) {
#pragma HLS PIPELINE II=1
	  lb[i][j] = 0.2 * (la[i][j] + la[i][j-1] + la[i][1+j] + la[1+i][j] + la[i-1][j]);
	}
      for (i = 1; i < n - 1; i++)
	for (j = 1; j < n - 1; j++) {
#pragma HLS PIPELINE II=1
	  la[i][j] = 0.2 * (lb[i][j] + lb[i][j-1] + lb[i][1+j] + lb[1+i][j] + lb[i-1][j]);
	}
    }

  // Copy results back to global memory.
  copy_out_i: for (i = 0; i < n; i++)
    copy_out_j: for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
      A[i][j] = la[i][j];
      B[i][j] = lb[i][j];
    }

}
}