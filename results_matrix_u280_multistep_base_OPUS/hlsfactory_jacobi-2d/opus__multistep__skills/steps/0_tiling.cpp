#include "jacobi-2d.h"
#include <cstring>

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

  // Local tile buffers staging the full working set
  static double A_local[N][N];
  static double B_local[N][N];

  // ---- LOAD phase ----
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
    {
#pragma HLS PIPELINE II=1
      A_local[i][j] = A[i][j];
      B_local[i][j] = B[i][j];
    }

  // ---- COMPUTE phase ----
  for (t = 0; t < tsteps; t++)
    {
      for (i = 1; i < n - 1; i++)
	for (j = 1; j < n - 1; j++)
	{
#pragma HLS PIPELINE II=1
	  B_local[i][j] = 0.2 * (A_local[i][j] + A_local[i][j-1] + A_local[i][1+j] + A_local[1+i][j] + A_local[i-1][j]);
	}
      for (i = 1; i < n - 1; i++)
	for (j = 1; j < n - 1; j++)
	{
#pragma HLS PIPELINE II=1
	  A_local[i][j] = 0.2 * (B_local[i][j] + B_local[i][j-1] + B_local[i][1+j] + B_local[1+i][j] + B_local[i-1][j]);
	}
    }

  // ---- STORE phase ----
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
    {
#pragma HLS PIPELINE II=1
      A[i][j] = A_local[i][j];
      B[i][j] = B_local[i][j];
    }

}