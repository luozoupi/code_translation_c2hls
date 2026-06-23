#include "gemver.h"
#include <string.h>


void kernel_gemver(
		   double alpha,
		   double beta,
		   double A[ N + 0][N + 0],
		   double u1[ N + 0],
		   double v1[ N + 0],
		   double u2[ N + 0],
		   double v2[ N + 0],
		   double w[ N + 0],
		   double x[ N + 0],
		   double y[ N + 0],
		   double z[ N + 0])
{
#pragma HLS INTERFACE m_axi port=A   offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=u1  offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=v1  offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=u2  offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=v2  offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=w   offset=slave bundle=gmem5
#pragma HLS INTERFACE m_axi port=x   offset=slave bundle=gmem6
#pragma HLS INTERFACE m_axi port=y   offset=slave bundle=gmem7
#pragma HLS INTERFACE m_axi port=z   offset=slave bundle=gmem8

#pragma HLS INTERFACE s_axilite port=alpha bundle=control
#pragma HLS INTERFACE s_axilite port=beta  bundle=control
#pragma HLS INTERFACE s_axilite port=A     bundle=control
#pragma HLS INTERFACE s_axilite port=u1    bundle=control
#pragma HLS INTERFACE s_axilite port=v1    bundle=control
#pragma HLS INTERFACE s_axilite port=u2    bundle=control
#pragma HLS INTERFACE s_axilite port=v2    bundle=control
#pragma HLS INTERFACE s_axilite port=w     bundle=control
#pragma HLS INTERFACE s_axilite port=x     bundle=control
#pragma HLS INTERFACE s_axilite port=y     bundle=control
#pragma HLS INTERFACE s_axilite port=z     bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;

  int i, j, ti;

  // Stage vectors into local buffers for fast reuse across the loop nests.
  double l_u1[N], l_v1[N], l_u2[N], l_v2[N];
  double l_x[N], l_y[N], l_z[N], l_w[N];
#pragma HLS ARRAY_PARTITION variable=l_v1 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_v2 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_x  cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_y  cyclic factor=8 dim=1

  // Local tile/buffer for the full matrix A (staged once, reused across phases).
  // Partition on both dimensions so row-wise (phase1/4) and column-wise (phase2)
  // accesses can be pipelined without bank conflicts.
  static double l_A[N][N];
#pragma HLS ARRAY_PARTITION variable=l_A cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_A cyclic factor=8 dim=1

  load_vecs: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
    l_u1[i] = u1[i];
    l_v1[i] = v1[i];
    l_u2[i] = u2[i];
    l_v2[i] = v2[i];
    l_x[i]  = x[i];
    l_y[i]  = y[i];
    l_z[i]  = z[i];
    l_w[i]  = w[i];
  }

  // ---- LOAD PHASE: stage matrix A into local buffer, tile by tile (row chunks) ----
  const int TILE = 256;
  load_A_outer: for (ti = 0; ti < n; ti += TILE) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=1
    int rows = (ti + TILE <= n) ? TILE : (n - ti);
    load_A_rows: for (int r = 0; r < rows; r++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=120
      memcpy(&l_A[ti + r][0], &A[ti + r][0], n * sizeof(double));
    }
  }

  // ---- COMPUTE PHASE (operates entirely on local buffers) ----

  // Phase 1: A = A + u1*v1^T + u2*v2^T
  phase1_i: for (i = 0; i < n; i++) {
    double u1i = l_u1[i];
    double u2i = l_u2[i];
    phase1_j: for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS DEPENDENCE variable=l_A inter false
      l_A[i][j] = l_A[i][j] + u1i * l_v1[j] + u2i * l_v2[j];
    }
  }

  // Phase 2: x = x + beta * A^T * y
  phase2_i: for (i = 0; i < n; i++) {
    double acc = l_x[i];
    phase2_j: for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS DEPENDENCE variable=l_A inter false
      acc += beta * l_A[j][i] * l_y[j];
    }
    l_x[i] = acc;
  }

  // Phase 3: x = x + z
  phase3: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
    l_x[i] = l_x[i] + l_z[i];
  }

  // Phase 4: w = w + alpha * A * x
  phase4_i: for (i = 0; i < n; i++) {
    double acc = l_w[i];
    phase4_j: for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS DEPENDENCE variable=l_A inter false
      acc += alpha * l_A[i][j] * l_x[j];
    }
    l_w[i] = acc;
  }

  // ---- STORE PHASE: write back modified matrix A (tile by tile) and result vectors ----
  store_A_outer: for (ti = 0; ti < n; ti += TILE) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=1
    int rows = (ti + TILE <= n) ? TILE : (n - ti);
    store_A_rows: for (int r = 0; r < rows; r++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=120
      memcpy(&A[ti + r][0], &l_A[ti + r][0], n * sizeof(double));
    }
  }

  store_out: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
    x[i] = l_x[i];
    w[i] = l_w[i];
  }
}