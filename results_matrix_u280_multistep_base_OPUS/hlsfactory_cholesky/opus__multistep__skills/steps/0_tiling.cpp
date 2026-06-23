#include "cholesky.h"
#include <cstring>

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
  double row_j[N];
#pragma HLS ARRAY_PARTITION variable=row_i cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=row_j cyclic factor=8 dim=1

  for (i = 0; i < n; i++) {

    // ---- LOAD phase: stage row i into local buffer ----
    load_row_i:
    for (k = 0; k < n; k++) {
#pragma HLS PIPELINE II=1
      row_i[k] = A[i][k];
    }

    // ---- COMPUTE phase (off-diagonal entries) ----
    for (j = 0; j < i; j++) {

      // LOAD phase: stage row j into local buffer
      load_row_j:
      for (k = 0; k < n; k++) {
#pragma HLS PIPELINE II=1
        row_j[k] = A[j][k];
      }

      // COMPUTE on local buffers
      double acc = row_i[j];
      compute_offdiag:
      for (k = 0; k < j; k++) {
#pragma HLS PIPELINE II=1
        acc -= row_i[k] * row_j[k];
      }

      double val = acc / row_j[j];
      row_i[j] = val;   // keep local copy consistent for later k reads
    }

    // ---- COMPUTE phase (diagonal entry) ----
    double diag = row_i[i];
    compute_diag:
    for (k = 0; k < i; k++) {
#pragma HLS PIPELINE II=1
      diag -= row_i[k] * row_i[k];
    }
    row_i[i] = sqrt(diag);

    // ---- STORE phase: write back updated row i ----
    store_row_i:
    for (k = 0; k <= i; k++) {
#pragma HLS PIPELINE II=1
      A[i][k] = row_i[k];
    }
  }

}