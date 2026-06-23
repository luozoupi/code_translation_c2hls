#include "gramschmidt.h"
#include <cstring>


// Load column j of A into buffer selected by flag
static void load_aj(double A[M + 0][N + 0],
                    double a_j_1[M], double a_j_2[M],
                    int j, bool flag)
{
#pragma HLS INLINE off
  const int m = M;
load_aj:
  for (int i = 0; i < m; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=60 max=60
    double v = A[i][j];
    if (flag) a_j_1[i] = v; else a_j_2[i] = v;
  }
}

// Compute (dot, update) and store column j using buffer selected by flag
static void compute_store_aj(double A[M + 0][N + 0],
                             double R[N + 0][N + 0],
                             double q_col[M],
                             double a_j_1[M], double a_j_2[M],
                             int k, int j, bool flag)
{
#pragma HLS INLINE off
  const int m = M;

  // ---- COMPUTE: dot product q_col . a_j ----
  double r_acc = 0.0;
dot_loop:
  for (int i = 0; i < m; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=60 max=60
    double v = flag ? a_j_1[i] : a_j_2[i];
    r_acc += q_col[i] * v;
  }
  R[k][j] = r_acc;

  // ---- COMPUTE: update column j in local buffer ----
  double r_val = r_acc;
update_loop:
  for (int i = 0; i < m; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=60 max=60
    double v = flag ? a_j_1[i] : a_j_2[i];
    v = v - q_col[i] * r_val;
    if (flag) a_j_1[i] = v; else a_j_2[i] = v;
  }

  // ---- STORE: write back updated column j of A ----
store_aj:
  for (int i = 0; i < m; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=60 max=60
    A[i][j] = flag ? a_j_1[i] : a_j_2[i];
  }
}


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
  // Double-buffered working columns for j_loop
  double a_j_1[M];
  double a_j_2[M];
#pragma HLS ARRAY_PARTITION variable=a_col cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=q_col cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=a_j_1 cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=a_j_2 cyclic factor=4 dim=1

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

      // ---- DOUBLE-BUFFERED j_loop ----
      // Overlap load of column (j+1) with compute/store of column j.
      int total = n - (k + 1);   // number of j iterations
      if (total > 0) {
        // Prologue: load first column into buffer 1 (flag=true)
        load_aj(A, a_j_1, a_j_2, k + 1, true);

      j_loop:
        for (int t = 0; t < total; t++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=80
          int j_cur = (k + 1) + t;
          bool flag_cur = ((t % 2) == 0); // buffer used by current compute

          // Prefetch next column into the OTHER buffer (overlaps compute)
          if (t + 1 < total) {
            int j_next = j_cur + 1;
            bool flag_next = !flag_cur;
            load_aj(A, a_j_1, a_j_2, j_next, flag_next);
          }

          // Compute + store current column from its buffer
          compute_store_aj(A, R, q_col, a_j_1, a_j_2, k, j_cur, flag_cur);
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