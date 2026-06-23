#include "doitgen.h"
#include <cstring>

extern "C" {

static void load_tile_fn(
    double A[NR][NQ][NP],
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
  load_tile_inner:
    for (int s = 0; s < np; s++) {
#pragma HLS PIPELINE II=1
      double val = A[rr][qq][s];
      if (flag == 0)
        A_in_1[i][s] = val;
      else
        A_in_2[i][s] = val;
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

void kernel_doitgen(
		    double A[ NR + 0][NQ + 0][NP + 0],
		    double C4[ NP + 0][NP + 0],
		    double sum[ NP + 0])
{
#pragma HLS INTERFACE m_axi port=A  offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=C4 offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=sum offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=A   bundle=control
#pragma HLS INTERFACE s_axilite port=C4  bundle=control
#pragma HLS INTERFACE s_axilite port=sum bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int nr = NR;
    const int nq = NQ;
    const int np = NP;

    // Tile size: number of (r,q) rows processed per tile.
    const int TILE = 256;

    int r, q, p, s, t, i;

    // Local buffer for C4 (reused across all r,q iterations).
    double C4_local[NP][NP];
#pragma HLS ARRAY_PARTITION variable=C4_local complete dim=1
#pragma HLS ARRAY_PARTITION variable=C4_local cyclic factor=2 dim=2

    // Double-buffered tile buffers (ping-pong).
    double A_in_1[TILE][NP];
#pragma HLS ARRAY_PARTITION variable=A_in_1 complete dim=2
    double A_in_2[TILE][NP];
#pragma HLS ARRAY_PARTITION variable=A_in_2 complete dim=2

    double A_out_1[TILE][NP];
#pragma HLS ARRAY_PARTITION variable=A_out_1 cyclic factor=2 dim=2
    double A_out_2[TILE][NP];
#pragma HLS ARRAY_PARTITION variable=A_out_2 cyclic factor=2 dim=2

    double sum_local[NP];
#pragma HLS ARRAY_PARTITION variable=sum_local complete dim=1

    // Load C4 once into local memory.
  load_c4_i:
    for (s = 0; s < np; s++) {
    load_c4_j:
      for (p = 0; p < np; p++) {
#pragma HLS PIPELINE II=1
        C4_local[s][p] = C4[s][p];
      }
    }

    const int total_rows = nr * nq; // total number of A[r][q] rows

    // Number of tiles.
    int num_tiles = (total_rows + TILE - 1) / TILE;

    // Helper lambdas inline via index recompute for store phase.
    // Pipelined software-pipeline across tiles:
    //   load tile k+1 overlaps compute of tile k.
    // We prefetch tile 0, then for each tile compute current while loading next.

    // Prefetch tile 0.
    int prev_t = 0;
    int prev_rows = (total_rows > TILE) ? TILE : total_rows;
    int prev_flag = 0;
    load_tile_fn(A, A_in_1, A_in_2, 0, prev_rows, nq, np, 0);

  tile_loop:
    for (int tk = 0; tk < num_tiles; tk++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=4

      int cur_flag = tk % 2;
      int cur_t = tk * TILE;
      int cur_rows = TILE;
      if (cur_t + cur_rows > total_rows)
        cur_rows = total_rows - cur_t;

      // ---------------- LOAD NEXT TILE (overlaps compute) ----------------
      int next_tk = tk + 1;
      if (next_tk < num_tiles) {
        int next_flag = next_tk % 2;
        int next_t = next_tk * TILE;
        int next_rows = TILE;
        if (next_t + next_rows > total_rows)
          next_rows = total_rows - next_t;
        load_tile_fn(A, A_in_1, A_in_2, next_t, next_rows, nq, np, next_flag);
      }

      // ---------------- COMPUTE CURRENT TILE ----------------
      compute_tile_fn(A_in_1, A_in_2, C4_local,
                      A_out_1, A_out_2, cur_rows, np, cur_flag);

      // ---------------- STORE PHASE ----------------
    store_tile:
      for (i = 0; i < cur_rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
        int idx = cur_t + i;
        int rr = idx / nq;
        int qq = idx % nq;
      store_tile_inner:
        for (p = 0; p < np; p++) {
#pragma HLS PIPELINE II=1
          double val = (cur_flag == 0) ? A_out_1[i][p] : A_out_2[i][p];
          A[rr][qq][p] = val;
          sum_local[p] = val;
        }
      }

      // Update sum with the last processed row of this tile.
    update_sum:
      for (p = 0; p < np; p++) {
#pragma HLS PIPELINE II=1
        sum[p] = sum_local[p];
      }
    }
}
}