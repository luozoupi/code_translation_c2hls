#include "trmm.h"

void kernel_trmm( 
		 double alpha,
		 double A[ M + 0][M + 0],
		 double B[ M + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=alpha bundle=control
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int m = M;
    const int n = N;

  int i, j, k;
  for (i = 0; i < m; i++)
     for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
        for (k = i+1; k < m; k++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=M
           B[i][j] += A[k][i] * B[k][j];
        }
        B[i][j] = alpha * B[i][j];
     }

}