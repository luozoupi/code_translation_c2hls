#include "symm.h"


void kernel_symm( 
		 double alpha,
		 double beta,
		 double C[ M + 0][N + 0],
		 double A[ M + 0][M + 0],
		 double B[ M + 0][N + 0])
{
#pragma HLS INLINE off

    const int m = M;
    const int n = N;

  int i, j, k;
  double temp2;

 loop_i: for (i = 0; i < m; i++)
  {
   loop_j: for (j = 0; j < n; j++ )
    {
#pragma HLS PIPELINE II=1
        temp2 = 0;
     loop_k: for (k = 0; k < i; k++) {
           C[k][j] += alpha*B[i][j] * A[i][k];
           temp2 += B[k][j] * A[i][k];
        }
        C[i][j] = beta * C[i][j] + alpha*B[i][j] * A[i][i] + alpha * temp2;
     }
  }

}

extern "C" {
void workload(
		 double alpha,
		 double beta,
		 double C[ M + 0][N + 0],
		 double A[ M + 0][M + 0],
		 double B[ M + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem2

#pragma HLS INTERFACE s_axilite port=alpha bundle=control
#pragma HLS INTERFACE s_axilite port=beta  bundle=control
#pragma HLS INTERFACE s_axilite port=C     bundle=control
#pragma HLS INTERFACE s_axilite port=A     bundle=control
#pragma HLS INTERFACE s_axilite port=B     bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    kernel_symm(alpha, beta, C, A, B);
}
}