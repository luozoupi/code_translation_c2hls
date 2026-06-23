#include "trisolv.h"


void kernel_trisolv(
		    double L[ N + 0][N + 0],
		    double x[ N + 0],
		    double b[ N + 0])
{
#pragma HLS INTERFACE m_axi port=L offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=L bundle=control
#pragma HLS INTERFACE s_axilite port=x bundle=control
#pragma HLS INTERFACE s_axilite port=b bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;

  int i, j;

  // Stage x into a local buffer for fast repeated access in the reduction
  double x_local[N];
#pragma HLS ARRAY_PARTITION variable=x_local cyclic factor=8 dim=1

  for (i = 0; i < n; i++)
    {
      double acc = b[i];
    inner_loop:
      for (j = 0; j < i; j++)
        {
#pragma HLS PIPELINE II=1
          acc -= L[i][j] * x_local[j];
        }
      x_local[i] = acc / L[i][i];
    }

  // Write back the result
write_back:
  for (i = 0; i < n; i++)
    {
#pragma HLS PIPELINE II=1
      x[i] = x_local[i];
    }

}