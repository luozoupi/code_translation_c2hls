#include "trisolv.h"

extern "C" {

void kernel_trisolv(
        double L[ N + 0][N + 0],
        double x[ N + 0],
        double b[ N + 0])
{
#pragma HLS INTERFACE m_axi port=L offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=L bundle=control
#pragma HLS INTERFACE s_axilite port=x bundle=control
#pragma HLS INTERFACE s_axilite port=b bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;
    const int TILE = 16; // tile size: process TILE rows at a time

    // Local buffers for b and x (full arrays, needed for dependency)
    double x_local[N];
    double b_local[N];

#pragma HLS ARRAY_PARTITION variable=x_local cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=b_local cyclic factor=8 dim=1

    // Double-buffered tile buffers for L rows (ping-pong)
    double L_tile_0[TILE][N];
    double L_tile_1[TILE][N];

#pragma HLS ARRAY_PARTITION variable=L_tile_0 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=L_tile_1 cyclic factor=8 dim=2

    // ---- LOAD PHASE: load b into local buffer ----
    load_b: for (int k = 0; k < n; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=N max=N
        b_local[k] = b[k];
    }

    // ---- INIT PHASE: initialize x_local to zero ----
    init_x: for (int k = 0; k < n; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=N max=N
        x_local[k] = 0.0;
    }

    // ---- COMPUTE PHASE: tiled triangular solve with double buffering ----
    // We preload tile 0 before the main loop, then in the loop:
    //   - compute tile (ping) while loading tile+1 into (pong)
    //   - swap ping/pong each iteration

    int num_tiles = (n + TILE - 1) / TILE;

    // Preload the first tile into L_tile_0 (buf=0)
    {
        int ti = 0;
        int tile_end = (ti + TILE < n) ? (ti + TILE) : n;
        int tile_rows = tile_end - ti;
        preload: for (int r = 0; r < tile_rows; r++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
            int i = ti + r;
            preload_row: for (int k = 0; k <= i; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=N
#pragma HLS DEPENDENCE variable=L_tile_0 inter false
                L_tile_0[r][k] = L[i][k];
            }
        }
    }

    // Main tile loop with double buffering
    tile_loop: for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
#pragma HLS LOOP_TRIPCOUNT min=N/16 max=N/16

        int ti = tile_idx * TILE;
        int tile_end = (ti + TILE < n) ? (ti + TILE) : n;
        int tile_rows = tile_end - ti;

        // Current buffer: buf=0 if tile_idx is even, buf=1 if odd
        int cur_buf = tile_idx % 2;   // buffer used for COMPUTE this iteration
        int nxt_buf = 1 - cur_buf;    // buffer used for LOAD (next tile) this iteration

        // LOAD NEXT TILE into nxt_buf (while compute runs on cur_buf)
        // Only load if there is a next tile
        int next_tile_idx = tile_idx + 1;
        int nti = next_tile_idx * TILE;
        if (next_tile_idx < num_tiles) {
            int ntile_end = (nti + TILE < n) ? (nti + TILE) : n;
            int ntile_rows = ntile_end - nti;

            load_tile: for (int r = 0; r < ntile_rows; r++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
                int i = nti + r;
                if (nxt_buf == 0) {
                    load_row_0: for (int k = 0; k <= i; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=N
#pragma HLS DEPENDENCE variable=L_tile_0 inter false
                        L_tile_0[r][k] = L[i][k];
                    }
                } else {
                    load_row_1: for (int k = 0; k <= i; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=N
#pragma HLS DEPENDENCE variable=L_tile_1 inter false
                        L_tile_1[r][k] = L[i][k];
                    }
                }
            }
        }

        // COMPUTE TILE from cur_buf
        compute_tile: for (int r = 0; r < tile_rows; r++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
            int i = ti + r;

            double xi = b_local[i];

            if (cur_buf == 0) {
                inner_0: for (int j = 0; j < i; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=0 max=N
#pragma HLS DEPENDENCE variable=x_local inter false
#pragma HLS DEPENDENCE variable=L_tile_0 inter false
                    xi -= L_tile_0[r][j] * x_local[j];
                }
                xi = xi / L_tile_0[r][i];
            } else {
                inner_1: for (int j = 0; j < i; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=0 max=N
#pragma HLS DEPENDENCE variable=x_local inter false
#pragma HLS DEPENDENCE variable=L_tile_1 inter false
                    xi -= L_tile_1[r][j] * x_local[j];
                }
                xi = xi / L_tile_1[r][i];
            }

            x_local[i] = xi;
        }
    }

    // ---- STORE PHASE: write x_local back to global memory ----
    store_x: for (int k = 0; k < n; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=N max=N
        x[k] = x_local[k];
    }
}

} // extern "C"