#include "atax.h"
#include <cstring>
#include <cstdint>

#define TILE 256
#define LARGE_BUS 512

// Number of 64-bit (double) elements per wide bus word
#define WIDE_FACTOR (LARGE_BUS / 64)

// Plain POD wide-bus word (512 bits = 8 doubles), portable substitute for ap_uint<512>
typedef struct {
  double v[WIDE_FACTOR];
} MARS_WIDE_BUS_TYPE;

// ---- Wide-bus helper functions (self-contained) ----
static void memcpy_wide_bus_read_float(double *local, MARS_WIDE_BUS_TYPE *bus,
                                       long elem_offset, int num_elems) {
  for (int e = 0; e < num_elems; e++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
#pragma HLS PIPELINE II=1
    long gidx = elem_offset + e;
    long word = gidx / WIDE_FACTOR;
    int sub = (int)(gidx % WIDE_FACTOR);
    local[e] = bus[word].v[sub];
  }
}

static void memcpy_wide_bus_write_float(MARS_WIDE_BUS_TYPE *bus, double *local,
                                        long elem_offset, int num_elems) {
  for (int e = 0; e < num_elems; e++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
#pragma HLS PIPELINE II=1
    long gidx = elem_offset + e;
    long word = gidx / WIDE_FACTOR;
    int sub = (int)(gidx % WIDE_FACTOR);
    bus[word].v[sub] = local[e];
  }
}

// Load a tile of A row into the selected buffer (flag picks buffer 1 or 2)
static void load_A_tile(MARS_WIDE_BUS_TYPE *A, double A_tile_1[TILE], double A_tile_2[TILE],
                        int i, int jt, int tile_size, int flag) {
  // Compute global element offset for A[i][jt]
  long offset = (long)i * N + jt;
  if (flag == 0) {
    memcpy_wide_bus_read_float(A_tile_1, A, offset, tile_size);
  } else {
    memcpy_wide_bus_read_float(A_tile_2, A, offset, tile_size);
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

// Internal wide-bus worker implementing the coalesced ATAX
static void atax_wide(MARS_WIDE_BUS_TYPE *A,
                      MARS_WIDE_BUS_TYPE *x,
                      MARS_WIDE_BUS_TYPE *y,
                      MARS_WIDE_BUS_TYPE *tmp) {
  const int m = M;
  const int n = N;

  int i, j;

  // Local buffers for the full x and y vectors (reused across all rows)
  double x_local[N];
  double y_local[N];
#pragma HLS ARRAY_PARTITION variable=x_local cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=y_local cyclic factor=4 dim=1

  // Local buffer for tmp results
  double tmp_local[M];
#pragma HLS ARRAY_PARTITION variable=tmp_local cyclic factor=4 dim=1

  // ---- LOAD x into local memory (coalesced) ----
  memcpy_wide_bus_read_float(x_local, x, 0, n);

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
      int num_tiles_tmp = (n + TILE - 1) / TILE;
      int first_tile_size = (TILE <= n) ? TILE : n;
      load_A_tile(A, A_tile_1, A_tile_2, i, 0, first_tile_size, 0);

TMP_TILE:
      for (int t = 0; t < num_tiles_tmp; t++) {
        int jt_cur = t * TILE;
        int tile_size = ((jt_cur + TILE) <= n) ? TILE : (n - jt_cur);
        int flag = t % 2;

        int t_next = t + 1;
        if (t_next < num_tiles_tmp) {
          int jt_next = t_next * TILE;
          int tile_size_next = ((jt_next + TILE) <= n) ? TILE : (n - jt_next);
          load_A_tile(A, A_tile_1, A_tile_2, i, jt_next, tile_size_next, t_next % 2);
        }

        compute_tmp_tile(A_tile_1, A_tile_2, x_local, jt_cur, tile_size, &tmp_acc, flag);
      }
      tmp_local[i] = tmp_acc;

      // --- Update y tile-by-tile with double buffering ---
      int num_tiles_y = (n + TILE - 1) / TILE;
      load_A_tile(A, A_tile_1, A_tile_2, i, 0, first_tile_size, 0);

Y_TILE:
      for (int t = 0; t < num_tiles_y; t++) {
        int jt_cur = t * TILE;
        int tile_size = ((jt_cur + TILE) <= n) ? TILE : (n - jt_cur);
        int flag = t % 2;

        int t_next = t + 1;
        if (t_next < num_tiles_y) {
          int jt_next = t_next * TILE;
          int tile_size_next = ((jt_next + TILE) <= n) ? TILE : (n - jt_next);
          load_A_tile(A, A_tile_1, A_tile_2, i, jt_next, tile_size_next, t_next % 2);
        }

        compute_y_tile(A_tile_1, A_tile_2, y_local, jt_cur, tile_size, tmp_acc, flag);
      }
    }

  // ---- STORE tmp back to global memory (coalesced) ----
  memcpy_wide_bus_write_float(tmp, tmp_local, 0, m);

  // ---- STORE y back to global memory (coalesced) ----
  memcpy_wide_bus_write_float(y, y_local, 0, n);
}

// Top function matches the header declaration exactly (double pointer types)
void kernel_atax(
		 double A[ M + 0][N + 0],
		 double x[ N + 0],
		 double y[ N + 0],
		 double tmp[ M + 0])
{
#pragma HLS INTERFACE m_axi port=A   offset=slave bundle=gmem0 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=x   offset=slave bundle=gmem1 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=y   offset=slave bundle=gmem2 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=tmp offset=slave bundle=gmem3 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE s_axilite port=A      bundle=control
#pragma HLS INTERFACE s_axilite port=x      bundle=control
#pragma HLS INTERFACE s_axilite port=y      bundle=control
#pragma HLS INTERFACE s_axilite port=tmp    bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  // Reinterpret the double arrays as wide-bus words for coalesced access
  atax_wide(reinterpret_cast<MARS_WIDE_BUS_TYPE *>(&A[0][0]),
            reinterpret_cast<MARS_WIDE_BUS_TYPE *>(x),
            reinterpret_cast<MARS_WIDE_BUS_TYPE *>(y),
            reinterpret_cast<MARS_WIDE_BUS_TYPE *>(tmp));
}