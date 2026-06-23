#include "cholesky.h"
#include <cstring>

// Load row j into the selected buffer
static void load_row_j(double A[N + 0][N + 0], double row_j_1[N], double row_j_2[N], int j, int n, int flag)
{
#pragma HLS INLINE off
  if (flag == 0) {
    load_j0:
    for (int k = 0; k < n; k++) {
#pragma HLS LOOP_TRIPCOUNT min=N max=N
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
      row_j_1[k] = A[j][k];
    }
  } else {
    load_j1:
    for (int k = 0; k < n; k++) {
#pragma HLS LOOP_TRIPCOUNT min=N max=N
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
      row_j_2[k] = A[j][k];
    }
  }
}

// Compute off-diagonal entry using the selected buffer
static double compute_offdiag(double row_i[N], double row_j_1[N], double row_j_2[N], int j, int flag)
{
#pragma HLS INLINE off
  double acc = row_i[j];
  if (flag == 0) {
    comp_j0:
    for (int k = 0; k < j; k++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=N
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
      acc -= row_i[k] * row_j_1[k];
    }
    return acc / row_j_1[j];
  } else {
    comp_j1:
    for (int k = 0; k < j; k++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=N
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
      acc -= row_i[k] * row_j_2[k];
    }
    return acc / row_j_2[j];
  }
}

void kernel_cholesky(
		     double A[ N + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;

  int i, j, k;

  // Local tile buffers for the row being processed and a helper row.
  double row_i[N];
  // Double-buffered helper rows (ping-pong).
  double row_j_1[N];
  double row_j_2[N];
#pragma HLS ARRAY_PARTITION variable=row_i cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=row_j_1 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=row_j_2 cyclic factor=8 dim=1

  for (i = 0; i < n; i++) {

    // ---- LOAD phase: stage row i into local buffer ----
    load_row_i:
    for (k = 0; k < n; k++) {
#pragma HLS LOOP_TRIPCOUNT min=N max=N
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
      row_i[k] = A[i][k];
    }

    // ---- COMPUTE phase (off-diagonal entries) with DOUBLE BUFFERING ----
    if (i > 0) {
      // Prologue: preload row j=0 into buffer 1
      load_row_j(A, row_j_1, row_j_2, 0, n, 0);

      for (j = 0; j < i; j++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=N
        int flag = j % 2;             // buffer holding current row j
        int next_flag = (j + 1) % 2;  // buffer to load row j+1 into

        // Load next row (j+1) into the other buffer while we compute on flag.
        if (j + 1 < i) {
          load_row_j(A, row_j_1, row_j_2, j + 1, n, next_flag);
        }

        // Compute off-diagonal using current buffer.
        double val = compute_offdiag(row_i, row_j_1, row_j_2, j, flag);
        row_i[j] = val;   // keep local copy consistent for later k reads
      }
    }

    // ---- COMPUTE phase (diagonal entry) ----
    double diag = row_i[i];
    compute_diag:
    for (k = 0; k < i; k++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=N
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
      diag -= row_i[k] * row_i[k];
    }
    row_i[i] = sqrt(diag);

    // ---- STORE phase: write back updated row i ----
    store_row_i:
    for (k = 0; k <= i; k++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=N
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
      A[i][k] = row_i[k];
    }
  }

}