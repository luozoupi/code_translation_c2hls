#include "jacobi-2d.h"

void kernel_jacobi_2d(

			    double A[ N + 0][N + 0],
			    double B[ N + 0][N + 0])
{
#pragma HLS INLINE off

    const int n = N;
    const int tsteps = TSTEPS;

  int t, i, j;

  for (t = 0; t < tsteps; t++)
    {
    stencil_B_i:
      for (i = 1; i < n - 1; i++)
      stencil_B_j:
	for (j = 1; j < n - 1; j++)
	  {
#pragma HLS PIPELINE II=1
	  B[i][j] = 0.2 * (A[i][j] + A[i][j-1] + A[i][1+j] + A[1+i][j] + A[i-1][j]);
	  }
    stencil_A_i:
      for (i = 1; i < n - 1; i++)
      stencil_A_j:
	for (j = 1; j < n - 1; j++)
	  {
#pragma HLS PIPELINE II=1
	  A[i][j] = 0.2 * (B[i][j] + B[i][j-1] + B[i][1+j] + B[1+i][j] + B[i-1][j]);
	  }
    }

}

extern "C" {
void workload(
			    double A[ N + 0][N + 0],
			    double B[ N + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    kernel_jacobi_2d(A, B);
}
}