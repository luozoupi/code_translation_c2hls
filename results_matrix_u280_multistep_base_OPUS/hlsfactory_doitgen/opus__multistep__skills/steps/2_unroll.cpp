#include "doitgen.h"
#include <cstring>

extern "C" {
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

    // Tile buffers: stage TILE rows of A locally for load/compute/store.
    double A_in[TILE][NP];
#pragma HLS ARRAY_PARTITION variable=A_in complete dim=2
    double A_out[TILE][NP];
#pragma HLS ARRAY_PARTITION variable=A_out cyclic factor=2 dim=2

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

    // Process all (r,q) rows in tiles.
  tile_loop:
    for (t = 0; t < total_rows; t += TILE) {

      int tile_rows = TILE;
      if (t + tile_rows > total_rows)
        tile_rows = total_rows - t;

      // ---------------- LOAD PHASE ----------------
    load_tile:
      for (i = 0; i < tile_rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
        int idx = t + i;
        int rr = idx / nq;
        int qq = idx % nq;
      load_tile_inner:
        for (s = 0; s < np; s++) {
#pragma HLS PIPELINE II=1
          A_in[i][s] = A[rr][qq][s];
        }
      }

      // ---------------- COMPUTE PHASE ----------------
    compute_tile:
      for (i = 0; i < tile_rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
      comp_p:
        for (p = 0; p < np; p++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=2
          double acc = 0.0;
        comp_s:
          for (s = 0; s < np; s++) {
#pragma HLS UNROLL
            acc += A_in[i][s] * C4_local[s][p];
          }
          A_out[i][p] = acc;
        }
      }

      // ---------------- STORE PHASE ----------------
    store_tile:
      for (i = 0; i < tile_rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
        int idx = t + i;
        int rr = idx / nq;
        int qq = idx % nq;
      store_tile_inner:
        for (p = 0; p < np; p++) {
#pragma HLS PIPELINE II=1
          double val = A_out[i][p];
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