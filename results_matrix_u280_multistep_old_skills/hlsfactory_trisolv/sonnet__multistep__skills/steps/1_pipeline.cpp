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

    // Local tile buffers for L rows (TILE rows x N columns)
    double L_tile[TILE][N];

#pragma HLS ARRAY_PARTITION variable=L_tile cyclic factor=8 dim=2

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

    // ---- COMPUTE PHASE: tiled triangular solve ----
    // Outer tile loop: process rows in tiles of size TILE
    tile_loop: for (int ti = 0; ti < n; ti += TILE) {
#pragma HLS LOOP_TRIPCOUNT min=N/16 max=N/16
        int tile_end = (ti + TILE < n) ? (ti + TILE) : n;
        int tile_rows = tile_end - ti;

        // LOAD TILE: load L rows [ti .. tile_end-1] into L_tile
        // Only load the relevant (lower triangular) columns for each row
        load_tile: for (int r = 0; r < tile_rows; r++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
            int i = ti + r;
            // Load columns 0..i (lower triangular part of row i)
            load_row: for (int k = 0; k <= i; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=N
#pragma HLS DEPENDENCE variable=L_tile inter false
                L_tile[r][k] = L[i][k];
            }
        }

        // COMPUTE TILE: solve x[i] for each row in this tile
        // Sequential over rows in tile due to carried dependency
        compute_tile: for (int r = 0; r < tile_rows; r++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
            int i = ti + r;

            double xi = b_local[i];

            // Inner loop: accumulate dot product using local L_tile row
            // x_local[j] for j < i are already fully written; no loop-carried dep here
            inner: for (int j = 0; j < i; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=0 max=N
#pragma HLS DEPENDENCE variable=x_local inter false
#pragma HLS DEPENDENCE variable=L_tile inter false
                xi -= L_tile[r][j] * x_local[j];
            }

            xi = xi / L_tile[r][i];
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