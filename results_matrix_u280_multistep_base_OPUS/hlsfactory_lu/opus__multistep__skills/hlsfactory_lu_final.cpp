#include "lu.h"
#include <cstring>
#include <cstdint>

// ---------------------------------------------------------------------------
// Wide-bus support (normally provided by common/mc.h). Defined inline here
// so the source compiles standalone without Xilinx headers.
// A 512-bit bus word holds 8 doubles.
// ---------------------------------------------------------------------------
#ifndef LARGE_BUS
#define LARGE_BUS 512
#endif

// Number of doubles that fit in one wide-bus word
#define DOUBLES_PER_BUS (LARGE_BUS / 64)

// 512-bit wide-bus word as a packed struct of doubles
typedef struct {
  double data[DOUBLES_PER_BUS];
} MARS_WIDE_BUS_TYPE;

// Read `num` doubles from wide-bus `bus` starting at element offset `elem_off`
static void memcpy_wide_bus_read_double(double *local,
                                        MARS_WIDE_BUS_TYPE *bus,
                                        long elem_off, int num)
{
  const long base_word = elem_off / DOUBLES_PER_BUS;
  int idx = 0;
READ_OUTER:
  for (int w = 0; idx < num; w++) {
    MARS_WIDE_BUS_TYPE tmp = bus[base_word + w];
  READ_INNER:
    for (int e = 0; e < DOUBLES_PER_BUS && idx < num; e++) {
#pragma HLS PIPELINE II=1
      local[idx++] = tmp.data[e];
    }
  }
}

// Write `num` doubles to wide-bus `bus` starting at element offset `elem_off`
static void memcpy_wide_bus_write_double(MARS_WIDE_BUS_TYPE *bus,
                                         double *local,
                                         long elem_off, int num)
{
  const long base_word = elem_off / DOUBLES_PER_BUS;
  int idx = 0;
WRITE_OUTER:
  for (int w = 0; idx < num; w++) {
    MARS_WIDE_BUS_TYPE tmp = bus[base_word + w];
  WRITE_INNER:
    for (int e = 0; e < DOUBLES_PER_BUS && idx < num; e++) {
#pragma HLS PIPELINE II=1
      tmp.data[e] = local[idx++];
    }
    bus[base_word + w] = tmp;
  }
}

// Load row A[i][*] into the selected row buffer (via wide bus)
static void load_row(MARS_WIDE_BUS_TYPE *A,
                     double row_i_0[N], double row_i_1[N],
                     int i, int flag)
{
  const int n = N;
  const long base = (long)i * n;  // element offset of row i
  if (flag == 0) {
    memcpy_wide_bus_read_double(row_i_0, A, base, n);
  } else {
    memcpy_wide_bus_read_double(row_i_1, A, base, n);
  }
}

// Compute on selected row buffer, using shared tile cache
static void compute_row(double row_i_0[N], double row_i_1[N],
                        double tile[N][N], int i, int flag)
{
  const int n = N;
  int j, k;

  if (flag == 0) {
  COMPUTE_LOWER_0:
    for (j = 0; j < i; j++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=N
      double acc = row_i_0[j];
    COMPUTE_LOWER_K_0:
      for (k = 0; k < j; k++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=N
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=2
        acc -= row_i_0[k] * tile[k][j];
      }
      row_i_0[j] = acc / tile[j][j];
    }
  COMPUTE_UPPER_0:
    for (j = i; j < n; j++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=N
      double acc = row_i_0[j];
    COMPUTE_UPPER_K_0:
      for (k = 0; k < i; k++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=N
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=2
        acc -= row_i_0[k] * tile[k][j];
      }
      row_i_0[j] = acc;
    }
  } else {
  COMPUTE_LOWER_1:
    for (j = 0; j < i; j++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=N
      double acc = row_i_1[j];
    COMPUTE_LOWER_K_1:
      for (k = 0; k < j; k++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=N
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=2
        acc -= row_i_1[k] * tile[k][j];
      }
      row_i_1[j] = acc / tile[j][j];
    }
  COMPUTE_UPPER_1:
    for (j = i; j < n; j++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=N
      double acc = row_i_1[j];
    COMPUTE_UPPER_K_1:
      for (k = 0; k < i; k++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=N
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=2
        acc -= row_i_1[k] * tile[k][j];
      }
      row_i_1[j] = acc;
    }
  }
}

// Store selected row buffer back to A[i][*] (via wide bus)
static void store_row(MARS_WIDE_BUS_TYPE *A,
                      double row_i_0[N], double row_i_1[N],
                      int i, int flag)
{
  const int n = N;
  const long base = (long)i * n;  // element offset of row i
  if (flag == 0) {
    memcpy_wide_bus_write_double(A, row_i_0, base, n);
  } else {
    memcpy_wide_bus_write_double(A, row_i_1, base, n);
  }
}

extern "C" {

void kernel_lu(
	       double A[ N + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;

  int i, j;

  // View the flat A matrix through the wide-bus type for coalesced accesses.
  MARS_WIDE_BUS_TYPE *A_bus = (MARS_WIDE_BUS_TYPE *)(&A[0][0]);

  // Double-buffered working row: ping-pong between two copies
  static double row_i_0[N];
  static double row_i_1[N];
  // Shared tile cache holding finalized previous rows
  static double tile[N][N];
#pragma HLS ARRAY_PARTITION variable=row_i_0 cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=row_i_1 cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=tile cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=tile cyclic factor=4 dim=2

  // Temporary buffer to bring finalized rows into the tile cache
  static double tile_load[N];
#pragma HLS ARRAY_PARTITION variable=tile_load cyclic factor=4 dim=1

  // Prologue: load first row into buffer 0
  load_row(A_bus, row_i_0, row_i_1, 0, 0);

  for (i = 0; i < n; i++) {

    int flag = i % 2;          // buffer currently being computed/stored
    int next_flag = (i + 1) % 2; // buffer used to prefetch next row

    // ---------- LOAD phase ----------
    // Prefetch next row A[i+1][*] into the OTHER buffer (overlaps with compute)
    if (i + 1 < n) {
      load_row(A_bus, row_i_0, row_i_1, i + 1, next_flag);
    }

    // Bring the newly finalized previous row A[i-1][*] into the tile cache
    if (i > 0) {
      memcpy_wide_bus_read_double(tile_load, A_bus, (long)(i - 1) * n, n);
    LOAD_TILE:
      for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
        tile[i - 1][j] = tile_load[j];
      }
    }

    // ---------- COMPUTE phase (on current buffer) ----------
    compute_row(row_i_0, row_i_1, tile, i, flag);

    // ---------- STORE phase ----------
    store_row(A_bus, row_i_0, row_i_1, i, flag);
  }
}
}