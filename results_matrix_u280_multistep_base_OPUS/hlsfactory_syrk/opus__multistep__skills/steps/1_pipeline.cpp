#include "syrk.h"

#define TILE 256

void kernel_syrk( 
		 double alpha,
		 double beta,
		 double C[ N + 0][N + 0],
		 double A[ N + 0][M + 0])
{
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=alpha bundle=control
#pragma HLS INTERFACE s_axilite port=beta bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;
    const int m = M;

  int i, j, k, jt, jj;

  // Stage A locally to enable reuse across the k/j loops (load phase for A).
  double A_local[N][M];
#pragma HLS ARRAY_PARTITION variable=A_local cyclic factor=8 dim=2

  for (i = 0; i < n; i++) {
    for (k = 0; k < m; k++) {
#pragma HLS PIPELINE II=1
      A_local[i][k] = A[i][k];
    }
  }

  for (i = 0; i < n; i++) {
    // Number of valid columns for this row (lower triangular part).
    int jmax = i + 1;

    // Process the C row in tiles of TILE columns.
    for (jt = 0; jt < jmax; jt += TILE) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=((N + TILE - 1) / TILE)

      int tile_end = jt + TILE;
      if (tile_end > jmax) tile_end = jmax;
      int tile_len = tile_end - jt;

      // ---- LOAD phase: stage this C tile into a local buffer ----
      double C_tile[TILE];
#pragma HLS ARRAY_PARTITION variable=C_tile cyclic factor=8 dim=1

      for (jj = 0; jj < tile_len; jj++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
        C_tile[jj] = C[i][jt + jj] * beta;
      }

      // ---- COMPUTE phase: operate purely on local buffers ----
      for (k = 0; k < m; k++) {
        double a_ik = A_local[i][k];
        for (jj = 0; jj < tile_len; jj++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
#pragma HLS DEPENDENCE variable=C_tile inter false
          C_tile[jj] += alpha * a_ik * A_local[jt + jj][k];
        }
      }

      // ---- STORE phase: write the tile back to global memory ----
      for (jj = 0; jj < tile_len; jj++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
        C[i][jt + jj] = C_tile[jj];
      }
    }
  }

}