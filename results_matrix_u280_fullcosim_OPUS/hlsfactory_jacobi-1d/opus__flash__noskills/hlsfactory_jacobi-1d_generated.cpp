#include "jacobi-1d.h"

extern "C" {
void kernel_jacobi_1d(
			    
			    double A[ N + 0],
			    double B[ N + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;
    const int tsteps = TSTEPS;

  int t, i;

  for (t = 0; t < tsteps; t++)
    {
      for (i = 1; i < n - 1; i++)
      {
#pragma HLS PIPELINE II=1
	B[i] = 0.33333 * (A[i-1] + A[i] + A[i + 1]);
      }
      for (i = 1; i < n - 1; i++)
      {
#pragma HLS PIPELINE II=1
	A[i] = 0.33333 * (B[i-1] + B[i] + B[i + 1]);
      }
    }

}
}