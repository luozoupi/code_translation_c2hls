#include "atax.h"
#include <cstring>

#define TILE 256

extern "C" {
void kernel_atax(
		 double A[ M + 0][N + 0],
		 double x[ N + 0],
		 double y[ N + 0],
		 double tmp[ M + 0])
{
#pragma HLS INTERFACE m_axi port=A   offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=x   offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=y   offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=tmp offset=slave bundle=gmem3
#pragma HLS INTERFACE s_axilite port=A      bundle=control
#pragma HLS INTERFACE s_axilite port=x      bundle=control
#pragma HLS INTERFACE s_axilite port=y      bundle=control
#pragma HLS INTERFACE s_axilite port=tmp    bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int m = M;
    const int n = N;

  int i, j, jt, jj;

  // Local buffers for the full x and y vectors (reused across all rows)
  double x_local[N];
  double y_local[N];
#pragma HLS ARRAY_PARTITION variable=x_local cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=y_local cyclic factor=4 dim=1

  // ---- LOAD x into local memory ----
LOAD_X:
  for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
    x_local[j] = x[j];
  }

  // ---- INIT y in local memory ----
INIT_Y:
  for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
    y_local[j] = 0;
  }

ROW_LOOP:
  for (i = 0; i < m; i++)
    {
      double tmp_acc = 0.0;

      // Process the columns of this row in tiles of TILE elements.
      // First pass: load tile of A row, compute tmp_acc (dot product).
      // Local tile buffer holds the reusable working set of the A row.
      double A_tile[TILE];
#pragma HLS ARRAY_PARTITION variable=A_tile cyclic factor=4 dim=1

      // --- Compute tmp_acc tile-by-tile (load + compute phases) ---
TMP_TILE:
      for (jt = 0; jt < n; jt += TILE) {
        int tile_size = ((jt + TILE) <= n) ? TILE : (n - jt);

        // LOAD phase: stage tile of A row into local buffer
LOAD_A_TMP:
        for (jj = 0; jj < tile_size; jj++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
#pragma HLS PIPELINE II=1
          A_tile[jj] = A[i][jt + jj];
        }

        // COMPUTE phase: dot product on local tile buffers
COMPUTE_TMP:
        for (jj = 0; jj < tile_size; jj++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
          tmp_acc = tmp_acc + A_tile[jj] * x_local[jt + jj];
        }
      }
      tmp[i] = tmp_acc;

      // --- Update y tile-by-tile (load + compute phases) ---
Y_TILE:
      for (jt = 0; jt < n; jt += TILE) {
        int tile_size = ((jt + TILE) <= n) ? TILE : (n - jt);

        // LOAD phase: stage tile of A row into local buffer
LOAD_A_Y:
        for (jj = 0; jj < tile_size; jj++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
#pragma HLS PIPELINE II=1
          A_tile[jj] = A[i][jt + jj];
        }

        // COMPUTE phase: accumulate into local y buffer
COMPUTE_Y:
        for (jj = 0; jj < tile_size; jj++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS DEPENDENCE variable=y_local inter false
          y_local[jt + jj] = y_local[jt + jj] + A_tile[jj] * tmp_acc;
        }
      }
    }

  // ---- STORE y back to global memory ----
STORE_Y:
  for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
    y[j] = y_local[j];
  }
}
}