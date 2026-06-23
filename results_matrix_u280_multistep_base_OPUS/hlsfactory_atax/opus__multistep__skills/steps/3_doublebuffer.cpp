#include "atax.h"
#include <cstring>

#define TILE 256

extern "C" {

// Load a tile of A row into the selected buffer (flag picks buffer 1 or 2)
static void load_A_tile(double A[M][N], double A_tile_1[TILE], double A_tile_2[TILE],
                        int i, int jt, int tile_size, int flag) {
  if (flag == 0) {
  LOAD_A_0:
    for (int jj = 0; jj < tile_size; jj++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
#pragma HLS PIPELINE II=1
      A_tile_1[jj] = A[i][jt + jj];
    }
  } else {
  LOAD_A_1:
    for (int jj = 0; jj < tile_size; jj++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
#pragma HLS PIPELINE II=1
      A_tile_2[jj] = A[i][jt + jj];
    }
  }
}

// Compute dot-product contribution from the selected buffer
static void compute_tmp_tile(double A_tile_1[TILE], double A_tile_2[TILE],
                             double x_local[N], int jt, int tile_size,
                             double *tmp_acc, int flag) {
  double acc = *tmp_acc;
  if (flag == 0) {
  COMPUTE_TMP_0:
    for (int jj = 0; jj < tile_size; jj++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
      acc = acc + A_tile_1[jj] * x_local[jt + jj];
    }
  } else {
  COMPUTE_TMP_1:
    for (int jj = 0; jj < tile_size; jj++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
      acc = acc + A_tile_2[jj] * x_local[jt + jj];
    }
  }
  *tmp_acc = acc;
}

// Compute y update from the selected buffer
static void compute_y_tile(double A_tile_1[TILE], double A_tile_2[TILE],
                           double y_local[N], int jt, int tile_size,
                           double tmp_acc, int flag) {
  if (flag == 0) {
  COMPUTE_Y_0:
    for (int jj = 0; jj < tile_size; jj++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS DEPENDENCE variable=y_local inter false
      y_local[jt + jj] = y_local[jt + jj] + A_tile_1[jj] * tmp_acc;
    }
  } else {
  COMPUTE_Y_1:
    for (int jj = 0; jj < tile_size; jj++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS DEPENDENCE variable=y_local inter false
      y_local[jt + jj] = y_local[jt + jj] + A_tile_2[jj] * tmp_acc;
    }
  }
}

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

  int i, j, jt;

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

  // Double-buffered tile storage for the A row
  double A_tile_1[TILE];
  double A_tile_2[TILE];
#pragma HLS ARRAY_PARTITION variable=A_tile_1 cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=A_tile_2 cyclic factor=4 dim=1

ROW_LOOP:
  for (i = 0; i < m; i++)
    {
      double tmp_acc = 0.0;

      // --- Compute tmp_acc tile-by-tile with double buffering ---
      // Prologue: load first tile
      int num_tiles_tmp = (n + TILE - 1) / TILE;
      int first_tile_size = (TILE <= n) ? TILE : n;
      load_A_tile(A, A_tile_1, A_tile_2, i, 0, first_tile_size, 0);

TMP_TILE:
      for (int t = 0; t < num_tiles_tmp; t++) {
        int jt_cur = t * TILE;
        int tile_size = ((jt_cur + TILE) <= n) ? TILE : (n - jt_cur);
        int flag = t % 2;

        // Load next tile (k+1) into the opposite buffer while computing tile k
        int t_next = t + 1;
        if (t_next < num_tiles_tmp) {
          int jt_next = t_next * TILE;
          int tile_size_next = ((jt_next + TILE) <= n) ? TILE : (n - jt_next);
          load_A_tile(A, A_tile_1, A_tile_2, i, jt_next, tile_size_next, t_next % 2);
        }

        // Compute on current tile from the current buffer
        compute_tmp_tile(A_tile_1, A_tile_2, x_local, jt_cur, tile_size, &tmp_acc, flag);
      }
      tmp[i] = tmp_acc;

      // --- Update y tile-by-tile with double buffering ---
      int num_tiles_y = (n + TILE - 1) / TILE;
      load_A_tile(A, A_tile_1, A_tile_2, i, 0, first_tile_size, 0);

Y_TILE:
      for (int t = 0; t < num_tiles_y; t++) {
        int jt_cur = t * TILE;
        int tile_size = ((jt_cur + TILE) <= n) ? TILE : (n - jt_cur);
        int flag = t % 2;

        // Load next tile (k+1) into the opposite buffer
        int t_next = t + 1;
        if (t_next < num_tiles_y) {
          int jt_next = t_next * TILE;
          int tile_size_next = ((jt_next + TILE) <= n) ? TILE : (n - jt_next);
          load_A_tile(A, A_tile_1, A_tile_2, i, jt_next, tile_size_next, t_next % 2);
        }

        // Compute y update on current tile
        compute_y_tile(A_tile_1, A_tile_2, y_local, jt_cur, tile_size, tmp_acc, flag);
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