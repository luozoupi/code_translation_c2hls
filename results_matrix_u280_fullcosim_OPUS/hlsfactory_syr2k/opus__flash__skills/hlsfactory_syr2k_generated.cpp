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
#pragma HLS INTERFACE s_axilite port=beta  bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;
    const int m = M;

    // Local buffers to stage the working set and enable parallel access.
    double C_buf[N][N];
    double A_buf[N][M];
    double B_buf[N][M];
#pragma HLS ARRAY_PARTITION variable=A_buf cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=B_buf cyclic factor=8 dim=2

    // Load A and B from global memory into local buffers.
  load_AB:
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < m; k++) {
#pragma HLS PIPELINE II=1
            A_buf[i][k] = A[i][k];
            B_buf[i][k] = B[i][k];
        }
    }

    // Load C from global memory into local buffer.
  load_C:
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            C_buf[i][j] = C[i][j];
        }
    }

  int i, j, k;
  compute_i:
  for (i = 0; i < n; i++) {
    scale_j:
    for (j = 0; j <= i; j++) {
#pragma HLS PIPELINE II=1
      C_buf[i][j] *= beta;
    }
    accum_k:
    for (k = 0; k < m; k++) {
      accum_j:
      for (j = 0; j <= i; j++)
	{
#pragma HLS PIPELINE II=1
	  C_buf[i][j] += A_buf[j][k]*alpha*B_buf[i][k] + B_buf[j][k]*alpha*A_buf[i][k];
	}
    }
  }

    // Store result C back to global memory.
  store_C:
    for (int ii = 0; ii < n; ii++) {
        for (int jj = 0; jj < n; jj++) {
#pragma HLS PIPELINE II=1
            C[ii][jj] = C_buf[ii][jj];
        }
    }
}