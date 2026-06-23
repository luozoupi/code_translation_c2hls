#include "gramschmidt.h"

extern "C" {
void kernel_gramschmidt(
			double A[ M + 0][N + 0],
			double R[ N + 0][N + 0],
			double Q[ M + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=R offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=Q offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=R bundle=control
#pragma HLS INTERFACE s_axilite port=Q bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int m = M;
    const int n = N;

  int i, j, k;

  double nrm;

  for (k = 0; k < n; k++)
    {
      nrm = 0.0;
    nrm_loop:
      for (i = 0; i < m; i++) {
#pragma HLS PIPELINE II=1
        nrm += A[i][k] * A[i][k];
      }
      R[k][k] = sqrt(nrm);
    q_loop:
      for (i = 0; i < m; i++) {
#pragma HLS PIPELINE II=1
        Q[i][k] = A[i][k] / R[k][k];
      }
    j_loop:
      for (j = k + 1; j < n; j++)
	{
	  R[k][j] = 0.0;
	  double r_acc = 0.0;
	r_loop:
	  for (i = 0; i < m; i++) {
#pragma HLS PIPELINE II=1
	    r_acc += Q[i][k] * A[i][j];
	  }
	  R[k][j] = r_acc;
	a_loop:
	  for (i = 0; i < m; i++) {
#pragma HLS PIPELINE II=1
	    A[i][j] = A[i][j] - Q[i][k] * r_acc;
	  }
	}
    }

}
}