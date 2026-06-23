#include "doitgen.h"
#include <cstring>

#ifndef LARGE_BUS
#define LARGE_BUS 512
#endif

// 512-bit wide bus type built from 8 x 64-bit words (no ap_int dependency).
#define DOUBLES_PER_BUS (LARGE_BUS / 64)  // 8 doubles per 512-bit word

struct wide_bus_t {
  unsigned long long w[DOUBLES_PER_BUS];
};

#define MARS_WIDE_BUS_TYPE wide_bus_t

// ---- Wide bus helper functions (inline, no external header) ----
static void memcpy_wide_bus_read_double(
    double *local, MARS_WIDE_BUS_TYPE *bus, long offset, long size_bytes)
{
  long start_word = offset / (LARGE_BUS / 8);
  long start_byte_in_word = offset % (LARGE_BUS / 8);
  int start_elem = (int)(start_byte_in_word / sizeof(double));
  int num_elems = (int)(size_bytes / sizeof(double));

  int produced = 0;
  long word = start_word;
  int elem = start_elem;
read_loop:
  while (produced < num_elems) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
#pragma HLS PIPELINE II=1
    MARS_WIDE_BUS_TYPE bw = bus[word];
    for (int k = 0; k < DOUBLES_PER_BUS; k++) {
#pragma HLS UNROLL
      if (k >= elem && produced + (k - elem) < num_elems) {
        unsigned long long u = bw.w[k];
        double d;
        std::memcpy(&d, &u, sizeof(double));
        local[produced + (k - elem)] = d;
      }
    }
    produced += (DOUBLES_PER_BUS - elem);
    elem = 0;
    word++;
  }
}

static void memcpy_wide_bus_write_double(
    MARS_WIDE_BUS_TYPE *bus, double *local, long offset, long size_bytes)
{
  long start_word = offset / (LARGE_BUS / 8);
  long start_byte_in_word = offset % (LARGE_BUS / 8);
  int start_elem = (int)(start_byte_in_word / sizeof(double));
  int num_elems = (int)(size_bytes / sizeof(double));

  int consumed = 0;
  long word = start_word;
  int elem = start_elem;
write_loop:
  while (consumed < num_elems) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
#pragma HLS PIPELINE II=1
    MARS_WIDE_BUS_TYPE bw = bus[word];
    for (int k = 0; k < DOUBLES_PER_BUS; k++) {
#pragma HLS UNROLL
      if (k >= elem && consumed + (k - elem) < num_elems) {
        double d = local[consumed + (k - elem)];
        unsigned long long u;
        std::memcpy(&u, &d, sizeof(double));
        bw.w[k] = u;
      }
    }
    bus[word] = bw;
    consumed += (DOUBLES_PER_BUS - elem);
    elem = 0;
    word++;
  }
}

static void load_tile_fn(
    MARS_WIDE_BUS_TYPE *A,
    double A_in_1[256][NP],
    double A_in_2[256][NP],
    int t, int tile_rows, int nq, int np, int flag)
{
load_tile:
  for (int i = 0; i < tile_rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
    int idx = t + i;
    int rr = idx / nq;
    int qq = idx % nq;
    long base = ((long)rr * nq + qq) * (long)np;
    if (flag == 0) {
      memcpy_wide_bus_read_double(A_in_1[i], A, base * sizeof(double), np * sizeof(double));
    } else {
      memcpy_wide_bus_read_double(A_in_2[i], A, base * sizeof(double), np * sizeof(double));
    }
  }
}

static void compute_tile_fn(
    double A_in_1[256][NP],
    double A_in_2[256][NP],
    double C4_local[NP][NP],
    double A_out_1[256][NP],
    double A_out_2[256][NP],
    int tile_rows, int np, int flag)
{
compute_tile:
  for (int i = 0; i < tile_rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
  comp_p:
    for (int p = 0; p < np; p++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=2
      double acc = 0.0;
    comp_s:
      for (int s = 0; s < np; s++) {
#pragma HLS UNROLL
        double a_val = (flag == 0) ? A_in_1[i][s] : A_in_2[i][s];
        acc += a_val * C4_local[s][p];
      }
      if (flag == 0)
        A_out_1[i][p] = acc;
      else
        A_out_2[i][p] = acc;
    }
  }
}

extern "C" {

void kernel_doitgen(
		    double A[ NR + 0][NQ + 0][NP + 0],
		    double C4[ NP + 0][NP + 0],
		    double sum[ NP + 0])
{
#pragma HLS INTERFACE m_axi port=A  offset=slave bundle=gmem0 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=C4 offset=slave bundle=gmem1 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=sum offset=slave bundle=gmem2 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE s_axilite port=A   bundle=control
#pragma HLS INTERFACE s_axilite port=C4  bundle=control
#pragma HLS INTERFACE s_axilite port=sum bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Reinterpret the global memory pointers as wide-bus (512-bit) pointers
    // for coalesced burst transfers.
    MARS_WIDE_BUS_TYPE *A_wide   = reinterpret_cast<MARS_WIDE_BUS_TYPE *>(A);
    MARS_WIDE_BUS_TYPE *C4_wide  = reinterpret_cast<MARS_WIDE_BUS_TYPE *>(C4);
    MARS_WIDE_BUS_TYPE *sum_wide = reinterpret_cast<MARS_WIDE_BUS_TYPE *>(sum);

    const int nr = NR;
    const int nq = NQ;
    const int np = NP;

    const int TILE = 256;

    int r, q, p, s, t, i;

    double C4_local[NP][NP];
#pragma HLS ARRAY_PARTITION variable=C4_local complete dim=1
#pragma HLS ARRAY_PARTITION variable=C4_local cyclic factor=2 dim=2

    double A_in_1[TILE][NP];
#pragma HLS ARRAY_PARTITION variable=A_in_1 complete dim=2
    double A_in_2[TILE][NP];
#pragma HLS ARRAY_PARTITION variable=A_in_2 complete dim=2

    double A_out_1[TILE][NP];
#pragma HLS ARRAY_PARTITION variable=A_out_1 cyclic factor=2 dim=2
    double A_out_2[TILE][NP];
#pragma HLS ARRAY_PARTITION variable=A_out_2 cyclic factor=2 dim=2

    double sum_local[NP];
#pragma HLS ARRAY_PARTITION variable=sum_local cyclic factor=2 dim=1

    double C4_row[NP];
#pragma HLS ARRAY_PARTITION variable=C4_row cyclic factor=2 dim=1
  load_c4_i:
    for (s = 0; s < np; s++) {
      long base = (long)s * np;
      memcpy_wide_bus_read_double(C4_row, C4_wide, base * sizeof(double), np * sizeof(double));
    load_c4_j:
      for (p = 0; p < np; p++) {
#pragma HLS PIPELINE II=1
        C4_local[s][p] = C4_row[p];
      }
    }

    const int total_rows = nr * nq;

    int num_tiles = (total_rows + TILE - 1) / TILE;

    int prev_rows = (total_rows > TILE) ? TILE : total_rows;
    load_tile_fn(A_wide, A_in_1, A_in_2, 0, prev_rows, nq, np, 0);

  tile_loop:
    for (int tk = 0; tk < num_tiles; tk++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=4

      int cur_flag = tk % 2;
      int cur_t = tk * TILE;
      int cur_rows = TILE;
      if (cur_t + cur_rows > total_rows)
        cur_rows = total_rows - cur_t;

      int next_tk = tk + 1;
      if (next_tk < num_tiles) {
        int next_flag = next_tk % 2;
        int next_t = next_tk * TILE;
        int next_rows = TILE;
        if (next_t + next_rows > total_rows)
          next_rows = total_rows - next_t;
        load_tile_fn(A_wide, A_in_1, A_in_2, next_t, next_rows, nq, np, next_flag);
      }

      compute_tile_fn(A_in_1, A_in_2, C4_local,
                      A_out_1, A_out_2, cur_rows, np, cur_flag);

    store_tile:
      for (i = 0; i < cur_rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
        int idx = cur_t + i;
        int rr = idx / nq;
        int qq = idx % nq;
        long base = ((long)rr * nq + qq) * (long)np;

      capture_sum:
        for (p = 0; p < np; p++) {
#pragma HLS PIPELINE II=1
          double val = (cur_flag == 0) ? A_out_1[i][p] : A_out_2[i][p];
          sum_local[p] = val;
        }

        if (cur_flag == 0) {
          memcpy_wide_bus_write_double(A_wide, A_out_1[i], base * sizeof(double), np * sizeof(double));
        } else {
          memcpy_wide_bus_write_double(A_wide, A_out_2[i], base * sizeof(double), np * sizeof(double));
        }
      }

      memcpy_wide_bus_write_double(sum_wide, sum_local, 0, np * sizeof(double));
    }
}
}