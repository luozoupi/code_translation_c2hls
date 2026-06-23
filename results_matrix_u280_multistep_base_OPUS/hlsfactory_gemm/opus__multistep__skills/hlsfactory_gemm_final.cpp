#include "gemm.h"
#include <string.h>

#ifndef LARGE_BUS
#define LARGE_BUS 512
#endif

// Number of double elements per wide bus word (512 / 64 = 8)
#define DOUBLES_PER_BUS (LARGE_BUS / 64)

// Wide bus word: a packed group of DOUBLES_PER_BUS doubles.
typedef struct {
    double data[DOUBLES_PER_BUS];
} MARS_WIDE_BUS_TYPE;

#define TILE_J 256

// ---- Wide-bus read helper for doubles ----
static void memcpy_wide_bus_read_double(
    double *local, MARS_WIDE_BUS_TYPE *bus,
    long byte_offset, long byte_len, int num)
{
    long elem_offset = byte_offset / (long)sizeof(double);
    int word_base = (int)(elem_offset / DOUBLES_PER_BUS);
    int lane0     = (int)(elem_offset % DOUBLES_PER_BUS);

    int idx = 0;
    int word = word_base;
    int lane = lane0;
  rd_loop:
    while (idx < num) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_J
        MARS_WIDE_BUS_TYPE w = bus[word];
      rd_lane:
        for (; lane < DOUBLES_PER_BUS && idx < num; lane++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=DOUBLES_PER_BUS
#pragma HLS PIPELINE II=1
            local[idx] = w.data[lane];
            idx++;
        }
        lane = 0;
        word++;
    }
}

// ---- Wide-bus write helper for doubles (read-modify-write at boundaries) ----
static void memcpy_wide_bus_write_double(
    MARS_WIDE_BUS_TYPE *bus, double *local,
    long byte_offset, long byte_len, int num)
{
    long elem_offset = byte_offset / (long)sizeof(double);
    int word_base = (int)(elem_offset / DOUBLES_PER_BUS);
    int lane0     = (int)(elem_offset % DOUBLES_PER_BUS);

    int idx = 0;
    int word = word_base;
    int lane = lane0;
  wr_loop:
    while (idx < num) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_J
        MARS_WIDE_BUS_TYPE w = bus[word];
      wr_lane:
        for (; lane < DOUBLES_PER_BUS && idx < num; lane++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=DOUBLES_PER_BUS
#pragma HLS PIPELINE II=1
            w.data[lane] = local[idx];
            idx++;
        }
        bus[word] = w;
        lane = 0;
        word++;
    }
}

// ---- LOAD phase: stage A row, C tile (beta-scaled), and B sub-block ----
static void load(
    int i, int jj, int tj,
    double beta,
    MARS_WIDE_BUS_TYPE *C,
    MARS_WIDE_BUS_TYPE *A,
    MARS_WIDE_BUS_TYPE *B,
    double A_row[NK],
    double C_tile[TILE_J],
    double B_tile[NK][TILE_J])
{
    const int nk = NK;
    int k, j;

    // ---- Load A row: contiguous segment A[i][0..nk-1] ----
    int a_base = i * NK;
    memcpy_wide_bus_read_double(
        A_row, A, (long)a_base * sizeof(double), nk * sizeof(double),
        nk);

    // ---- Load C tile (then beta-scale) ----
    int c_base = i * NJ + jj;
    memcpy_wide_bus_read_double(
        C_tile, C, (long)c_base * sizeof(double), tj * sizeof(double),
        tj);
  scale_C:
    for (j = 0; j < tj; j++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_J
#pragma HLS PIPELINE II=1
        C_tile[j] = C_tile[j] * beta;
    }

    // ---- Load B sub-block row by row ----
  load_B:
    for (k = 0; k < nk; k++) {
        int b_base = k * NJ + jj;
        memcpy_wide_bus_read_double(
            B_tile[k], B, (long)b_base * sizeof(double), tj * sizeof(double),
            tj);
    }
}

// ---- COMPUTE + STORE phase: accumulate over k, then write back ----
static void compute(
    int i, int jj, int tj,
    double alpha,
    MARS_WIDE_BUS_TYPE *C,
    double A_row[NK],
    double C_tile[TILE_J],
    double B_tile[NK][TILE_J])
{
    const int nk = NK;
    int k, j;

  compute_k:
    for (k = 0; k < nk; k++) {
        double a_val = alpha * A_row[k];
      compute_j:
        for (j = 0; j < tj; j++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_J
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS DEPENDENCE variable=C_tile inter false
            C_tile[j] += a_val * B_tile[k][j];
        }
    }

    // ---- Store C tile back ----
    int c_base = i * NJ + jj;
    memcpy_wide_bus_write_double(
        C, C_tile, (long)c_base * sizeof(double), tj * sizeof(double),
        tj);
}

extern "C" {
void kernel_gemm(
		 double alpha,
		 double beta,
		 double C[ NI + 0][NJ + 0],
		 double A[ NI + 0][NK + 0],
		 double B[ NK + 0][NJ + 0])
{
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem0 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem1 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem2 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE s_axilite port=alpha bundle=control
#pragma HLS INTERFACE s_axilite port=beta bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int ni = NI;
    const int nj = NJ;

  // Reinterpret the flat global arrays as wide-bus words for coalesced access.
  MARS_WIDE_BUS_TYPE *C_bus = (MARS_WIDE_BUS_TYPE *)C;
  MARS_WIDE_BUS_TYPE *A_bus = (MARS_WIDE_BUS_TYPE *)A;
  MARS_WIDE_BUS_TYPE *B_bus = (MARS_WIDE_BUS_TYPE *)B;

  int i, jj;

  // ---- DOUBLE-BUFFERED local tile buffers (ping-pong pair) ----
  double A_row_1[NK];
  double A_row_2[NK];
  double C_tile_1[TILE_J];
  double C_tile_2[TILE_J];
  double B_tile_1[NK][TILE_J];
  double B_tile_2[NK][TILE_J];

  // Partition both buffer sets along the j dimension to match the unroll
  // factor of the compute_j loop so parallel iterations access distinct banks.
#pragma HLS ARRAY_PARTITION variable=B_tile_1 cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=B_tile_2 cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=C_tile_1 cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=C_tile_2 cyclic factor=4 dim=1

  // ---- Prologue: load the first tile ----
  int first_i = 0;
  int first_jj = 0;
  int first_tj = (nj < TILE_J) ? nj : TILE_J;
  load(first_i, first_jj, first_tj, beta, C_bus, A_bus, B_bus,
       A_row_1, C_tile_1, B_tile_1);

  int flag = 0;          // which buffer set currently holds loaded data
  int cur_i = first_i;
  int cur_jj = first_jj;
  int cur_tj = first_tj;

  // ---- Steady state: overlap load(next) with compute(current) ----
  for (i = 0; i < ni; i++) {
    for (jj = 0; jj < nj; jj += TILE_J) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=NI

      // Skip the very first tile (already prologue-loaded as current).
      if (i == 0 && jj == 0) continue;

      int j_end = jj + TILE_J;
      if (j_end > nj) j_end = nj;
      int tj = j_end - jj;

      // Load the next tile into the OTHER buffer set while computing current.
      if (flag == 0) {
        load(i, jj, tj, beta, C_bus, A_bus, B_bus,
             A_row_2, C_tile_2, B_tile_2);
        compute(cur_i, cur_jj, cur_tj, alpha, C_bus,
                A_row_1, C_tile_1, B_tile_1);
      } else {
        load(i, jj, tj, beta, C_bus, A_bus, B_bus,
             A_row_1, C_tile_1, B_tile_1);
        compute(cur_i, cur_jj, cur_tj, alpha, C_bus,
                A_row_2, C_tile_2, B_tile_2);
      }

      // Advance: the freshly loaded tile becomes the current tile.
      flag = 1 - flag;
      cur_i = i;
      cur_jj = jj;
      cur_tj = tj;
    }
  }

  // ---- Epilogue: compute the last loaded tile ----
  if (flag == 0) {
    compute(cur_i, cur_jj, cur_tj, alpha, C_bus,
            A_row_1, C_tile_1, B_tile_1);
  } else {
    compute(cur_i, cur_jj, cur_tj, alpha, C_bus,
            A_row_2, C_tile_2, B_tile_2);
  }
}
}