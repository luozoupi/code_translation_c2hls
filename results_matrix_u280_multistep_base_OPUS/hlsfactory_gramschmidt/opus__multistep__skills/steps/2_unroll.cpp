#include "gramschmidt.h"
#include <cstring>


void kernel_gramschmidt( 
			double A[ M + 0][N + 0],
			double R[ N + 0][N + 0],
			double Q[ M + 0][N + 0])
{
#pragma HLS INLINE off

    const int m = M;
    const int n = N;

  int i, j, k;

  double nrm;

  // Local tile buffers for column-wise reuse
  double a_col[M];   // staged column k of A
  double q_col[M];   // staged column k of Q
  double a_j[M];     // staged column j of A (working column)
#pragma HLS ARRAY_PARTITION variable=a_col cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=q_col cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=a_j   cyclic factor=4 dim=1

  for (k = 0; k < n; k++)
    {
      // ---- LOAD: stage column k of A into local buffer ----
    load_acol:
      for (i = 0; i < m; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=60 max=60
        a_col[i] = A[i][k];
      }

      // ---- COMPUTE: norm of column k from local buffer ----
      nrm = 0.0;
    nrm_loop:
      for (i = 0; i < m; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=60 max=60
        nrm += a_col[i] * a_col[i];
      }

      double r_kk = sqrt(nrm);
      R[k][k] = r_kk;

      // ---- COMPUTE: normalize column k into local q_col ----
    q_loop:
      for (i = 0; i < m; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=60 max=60
        q_col[i] = a_col[i] / r_kk;
      }

      // ---- STORE: write back Q column k ----
    store_qcol:
      for (i = 0; i < m; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=60 max=60
        Q[i][k] = q_col[i];
      }

    j_loop:
      for (j = k + 1; j < n; j++)
	{
#pragma HLS LOOP_TRIPCOUNT min=1 max=80
	  // ---- LOAD: stage column j of A into local buffer ----
	load_aj:
	  for (i = 0; i < m; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=60 max=60
	    a_j[i] = A[i][j];
	  }

	  // ---- COMPUTE: dot product q_col . a_j ----
	  double r_acc = 0.0;
	dot_loop:
	  for (i = 0; i < m; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=60 max=60
	    r_acc += q_col[i] * a_j[i];
	  }
	  R[k][j] = r_acc;

	  // ---- COMPUTE: update column j in local buffer ----
	  double r_val = r_acc;
	update_loop:
	  for (i = 0; i < m; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=60 max=60
	    a_j[i] = a_j[i] - q_col[i] * r_val;
	  }

	  // ---- STORE: write back updated column j of A ----
	store_aj:
	  for (i = 0; i < m; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=60 max=60
	    A[i][j] = a_j[i];
	  }
	}
    }

}

extern "C" {
void workload(
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

    kernel_gramschmidt(A, R, Q);
}
}