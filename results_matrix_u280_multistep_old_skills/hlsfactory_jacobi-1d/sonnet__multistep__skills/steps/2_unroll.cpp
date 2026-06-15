#include "jacobi-1d.h"
#include <string.h>

#define TILE_SIZE 64

extern "C" {

void kernel_jacobi_1d(
    double A[N + 0],
    double B[N + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local tile buffers for A and B
    double lA[N];
    double lB[N];

    // Increase partitioning factor to match unroll factor of 4
#pragma HLS ARRAY_PARTITION variable=lA cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=lB cyclic factor=4 dim=1

    const int n      = N;
    const int tsteps = TSTEPS;

    // ---------------------------------------------------------------
    // LOAD phase: bring A and B from global memory in tiles
    // ---------------------------------------------------------------
    load_A_tiles: for (int tile_start = 0; tile_start < n; tile_start += TILE_SIZE) {
        int tile_end = (tile_start + TILE_SIZE < n) ? tile_start + TILE_SIZE : n;
        load_A_inner: for (int i = tile_start; i < tile_end; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=TILE_SIZE
#pragma HLS DEPENDENCE variable=lA inter false
            lA[i] = A[i];
        }
    }

    load_B_tiles: for (int tile_start = 0; tile_start < n; tile_start += TILE_SIZE) {
        int tile_end = (tile_start + TILE_SIZE < n) ? tile_start + TILE_SIZE : n;
        load_B_inner: for (int i = tile_start; i < tile_end; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=TILE_SIZE
#pragma HLS DEPENDENCE variable=lB inter false
            lB[i] = B[i];
        }
    }

    // ---------------------------------------------------------------
    // COMPUTE phase: run time steps entirely on local buffers
    // ---------------------------------------------------------------
    for (int t = 0; t < tsteps; t++) {
#pragma HLS LOOP_TRIPCOUNT min=TSTEPS max=TSTEPS

        // Update B from A, tiled over the interior
        update_B_tiles: for (int tile_start = 1; tile_start < n - 1; tile_start += TILE_SIZE) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=(N/TILE_SIZE+1)
            int tile_end = (tile_start + TILE_SIZE < n - 1) ? tile_start + TILE_SIZE : n - 1;

            // Stage a local tile for reading A neighbors
            double tileA[TILE_SIZE + 2];
#pragma HLS ARRAY_PARTITION variable=tileA complete dim=1

            // Load tile of A (with left and right halo) into local tile buffer
            load_tileA: for (int i = 0; i < (tile_end - tile_start + 2); i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=(TILE_SIZE+2)
#pragma HLS DEPENDENCE variable=tileA inter false
                tileA[i] = lA[tile_start - 1 + i];
            }

            // Compute B updates using staged tile - unroll for data parallelism
            compute_B: for (int i = tile_start; i < tile_end; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=TILE_SIZE
#pragma HLS DEPENDENCE variable=lB inter false
                int li = i - tile_start; // local index within tile (offset by 1 for halo)
                lB[i] = 0.33333 * (tileA[li] + tileA[li + 1] + tileA[li + 2]);
            }
        }

        // Update A from B, tiled over the interior
        update_A_tiles: for (int tile_start = 1; tile_start < n - 1; tile_start += TILE_SIZE) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=(N/TILE_SIZE+1)
            int tile_end = (tile_start + TILE_SIZE < n - 1) ? tile_start + TILE_SIZE : n - 1;

            // Stage a local tile for reading B neighbors
            double tileB[TILE_SIZE + 2];
#pragma HLS ARRAY_PARTITION variable=tileB complete dim=1

            // Load tile of B (with left and right halo) into local tile buffer
            load_tileB: for (int i = 0; i < (tile_end - tile_start + 2); i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=(TILE_SIZE+2)
#pragma HLS DEPENDENCE variable=tileB inter false
                tileB[i] = lB[tile_start - 1 + i];
            }

            // Compute A updates using staged tile - unroll for data parallelism
            compute_A: for (int i = tile_start; i < tile_end; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=TILE_SIZE
#pragma HLS DEPENDENCE variable=lA inter false
                int li = i - tile_start;
                lA[i] = 0.33333 * (tileB[li] + tileB[li + 1] + tileB[li + 2]);
            }
        }
    }

    // ---------------------------------------------------------------
    // STORE phase: write local buffers back to global memory in tiles
    // ---------------------------------------------------------------
    store_A_tiles: for (int tile_start = 0; tile_start < n; tile_start += TILE_SIZE) {
        int tile_end = (tile_start + TILE_SIZE < n) ? tile_start + TILE_SIZE : n;
        store_A_inner: for (int i = tile_start; i < tile_end; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=TILE_SIZE
#pragma HLS DEPENDENCE variable=lA inter false
            A[i] = lA[i];
        }
    }

    store_B_tiles: for (int tile_start = 0; tile_start < n; tile_start += TILE_SIZE) {
        int tile_end = (tile_start + TILE_SIZE < n) ? tile_start + TILE_SIZE : n;
        store_B_inner: for (int i = tile_start; i < tile_end; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=TILE_SIZE
#pragma HLS DEPENDENCE variable=lB inter false
            B[i] = lB[i];
        }
    }
}

} // extern "C"