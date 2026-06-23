#include "gesummv.h"


void kernel_gesummv(
		    double alpha,
		    double beta,
		    double A[ N + 0][N + 0],
		    double B[ N + 0][N + 0],
		    double tmp[ N + 0],
		    double x[ N + 0],
		    double y[ N + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=tmp offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=y offset=slave bundle=gmem2

#pragma HLS INTERFACE s_axilite port=alpha bundle=control
#pragma HLS INTERFACE s_axilite port=beta bundle=control
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=tmp bundle=control
#pragma HLS INTERFACE s_axilite port=x bundle=control
#pragma HLS INTERFACE s_axilite port=y bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;

  int i, j;

  // Stage x locally for reuse across all rows
  double x_local[N];
#pragma HLS ARRAY_PARTITION variable=x_local cyclic factor=8 dim=1
  for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
    x_local[j] = x[j];
  }

  for (i = 0; i < n; i++)
    {
      double acc_tmp = 0.0;
      double acc_y = 0.0;

      // Stage one row of A and B locally for reuse / parallel access
      double A_row[N];
      double B_row[N];
#pragma HLS ARRAY_PARTITION variable=A_row cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=B_row cyclic factor=8 dim=1

      for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
        A_row[j] = A[i][j];
        B_row[j] = B[i][j];
      }

      for (j = 0; j < n; j++)
	{
#pragma HLS PIPELINE II=1
	  acc_tmp = A_row[j] * x_local[j] + acc_tmp;
	  acc_y = B_row[j] * x_local[j] + acc_y;
	}

      tmp[i] = acc_tmp;
      y[i] = alpha * acc_tmp + beta * acc_y;
    }

}