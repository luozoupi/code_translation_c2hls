#include "jacobi-1d.h"
#include <string.h>

#define TILE_SIZE 64

extern "C" {

void kernel_jacobi_1d(
    double A[N + 0],
    double B[N + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local tile buffers for A and B
    double lA[N];
    double lB[N];

    // Cyclic partitioning factor=8 to match 512-bit wide bus (8 doubles per word)
#pragma HLS ARRAY_PARTITION variable=lA cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=lB cyclic factor=8 dim=1

    const int n      = N;
    const int tsteps = TSTEPS;

    // ---------------------------------------------------------------
    // LOAD phase: bring A and B from global memory in tiles
    // ---------------------------------------------------------------
    load_A_tiles: for (int tile_start = 0; tile_start < n; tile_start += TILE_SIZE) {
        int tile_end = (tile_start + TILE_SIZE < n) ? tile_start + TILE_SIZE : n;
        load_A_inner: for (int i = tile_start; i < tile_end; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=TILE_SIZE
#pragma HLS DEPENDENCE variable=lA inter false
            lA[i] = A[i];
        }
    }

    load_B_tiles: for (int tile_start = 0; tile_start < n; tile_start += TILE_SIZE) {
        int tile_end = (tile_start + TILE_SIZE < n) ? tile_start + TILE_SIZE : n;
        load_B_inner: for (int i = tile_start; i < tile_end; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=TILE_SIZE
#pragma HLS DEPENDENCE variable=lB inter false
            lB[i] = B[i];
        }
    }

    // ---------------------------------------------------------------
    // COMPUTE phase: run time steps entirely on local buffers
    // Double-buffered tileA and tileB ping-pong buffers
    // ---------------------------------------------------------------

    // Double-buffered staging arrays for tileA (used in B update)
    double tileA_0[TILE_SIZE + 2];
    double tileA_1[TILE_SIZE + 2];
#pragma HLS ARRAY_PARTITION variable=tileA_0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=tileA_1 complete dim=1

    // Double-buffered staging arrays for tileB (used in A update)
    double tileB_0[TILE_SIZE + 2];
    double tileB_1[TILE_SIZE + 2];
#pragma HLS ARRAY_PARTITION variable=tileB_0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=tileB_1 complete dim=1

    for (int t = 0; t < tsteps; t++) {
#pragma HLS LOOP_TRIPCOUNT min=TSTEPS max=TSTEPS

        // ===========================================================
        // Update B from A, double-buffered tiling over interior
        // ===========================================================
        {
            // Pre-load tile 0 into buffer 0 (ping)
            {
                int tile_start = 1;
                int tile_end = (tile_start + TILE_SIZE < n - 1) ? tile_start + TILE_SIZE : n - 1;
                int len = tile_end - tile_start + 2;
                load_tileA_init: for (int i = 0; i < len; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=(TILE_SIZE+2)
#pragma HLS DEPENDENCE variable=tileA_0 inter false
                    tileA_0[i] = lA[tile_start - 1 + i];
                }
            }

            int tile_idx = 0;
            update_B_tiles: for (int tile_start = 1; tile_start < n - 1; tile_start += TILE_SIZE) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=(N/TILE_SIZE+1)
                int tile_end = (tile_start + TILE_SIZE < n - 1) ? tile_start + TILE_SIZE : n - 1;
                int next_tile_start = tile_start + TILE_SIZE;
                int buf_sel = tile_idx % 2; // 0 = use tileA_0, 1 = use tileA_1

                // Pre-load next tile into the other buffer (if next tile exists)
                if (next_tile_start < n - 1) {
                    int next_tile_end = (next_tile_start + TILE_SIZE < n - 1) ? next_tile_start + TILE_SIZE : n - 1;
                    int next_len = next_tile_end - next_tile_start + 2;
                    if (buf_sel == 0) {
                        // Load next tile into tileA_1 while computing from tileA_0
                        prefetch_tileA_1: for (int i = 0; i < next_len; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=(TILE_SIZE+2)
#pragma HLS DEPENDENCE variable=tileA_1 inter false
                            tileA_1[i] = lA[next_tile_start - 1 + i];
                        }
                    } else {
                        // Load next tile into tileA_0 while computing from tileA_1
                        prefetch_tileA_0: for (int i = 0; i < next_len; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=(TILE_SIZE+2)
#pragma HLS DEPENDENCE variable=tileA_0 inter false
                            tileA_0[i] = lA[next_tile_start - 1 + i];
                        }
                    }
                }

                // Compute B updates from the current buffer
                if (buf_sel == 0) {
                    compute_B_0: for (int i = tile_start; i < tile_end; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=TILE_SIZE
#pragma HLS DEPENDENCE variable=lB inter false
                        int li = i - tile_start;
                        lB[i] = 0.33333 * (tileA_0[li] + tileA_0[li + 1] + tileA_0[li + 2]);
                    }
                } else {
                    compute_B_1: for (int i = tile_start; i < tile_end; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=TILE_SIZE
#pragma HLS DEPENDENCE variable=lB inter false
                        int li = i - tile_start;
                        lB[i] = 0.33333 * (tileA_1[li] + tileA_1[li + 1] + tileA_1[li + 2]);
                    }
                }

                tile_idx++;
            }
        }

        // ===========================================================
        // Update A from B, double-buffered tiling over interior
        // ===========================================================
        {
            // Pre-load tile 0 into buffer 0 (ping)
            {
                int tile_start = 1;
                int tile_end = (tile_start + TILE_SIZE < n - 1) ? tile_start + TILE_SIZE : n - 1;
                int len = tile_end - tile_start + 2;
                load_tileB_init: for (int i = 0; i < len; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=(TILE_SIZE+2)
#pragma HLS DEPENDENCE variable=tileB_0 inter false
                    tileB_0[i] = lB[tile_start - 1 + i];
                }
            }

            int tile_idx = 0;
            update_A_tiles: for (int tile_start = 1; tile_start < n - 1; tile_start += TILE_SIZE) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=(N/TILE_SIZE+1)
                int tile_end = (tile_start + TILE_SIZE < n - 1) ? tile_start + TILE_SIZE : n - 1;
                int next_tile_start = tile_start + TILE_SIZE;
                int buf_sel = tile_idx % 2;

                // Pre-load next tile into the other buffer (if next tile exists)
                if (next_tile_start < n - 1) {
                    int next_tile_end = (next_tile_start + TILE_SIZE < n - 1) ? next_tile_start + TILE_SIZE : n - 1;
                    int next_len = next_tile_end - next_tile_start + 2;
                    if (buf_sel == 0) {
                        prefetch_tileB_1: for (int i = 0; i < next_len; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=(TILE_SIZE+2)
#pragma HLS DEPENDENCE variable=tileB_1 inter false
                            tileB_1[i] = lB[next_tile_start - 1 + i];
                        }
                    } else {
                        prefetch_tileB_0: for (int i = 0; i < next_len; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=(TILE_SIZE+2)
#pragma HLS DEPENDENCE variable=tileB_0 inter false
                            tileB_0[i] = lB[next_tile_start - 1 + i];
                        }
                    }
                }

                // Compute A updates from the current buffer
                if (buf_sel == 0) {
                    compute_A_0: for (int i = tile_start; i < tile_end; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=TILE_SIZE
#pragma HLS DEPENDENCE variable=lA inter false
                        int li = i - tile_start;
                        lA[i] = 0.33333 * (tileB_0[li] + tileB_0[li + 1] + tileB_0[li + 2]);
                    }
                } else {
                    compute_A_1: for (int i = tile_start; i < tile_end; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=TILE_SIZE
#pragma HLS DEPENDENCE variable=lA inter false
                        int li = i - tile_start;
                        lA[i] = 0.33333 * (tileB_1[li] + tileB_1[li + 1] + tileB_1[li + 2]);
                    }
                }

                tile_idx++;
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
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=TILE_SIZE
#pragma HLS DEPENDENCE variable=lA inter false
            A[i] = lA[i];
        }
    }

    store_B_tiles: for (int tile_start = 0; tile_start < n; tile_start += TILE_SIZE) {
        int tile_end = (tile_start + TILE_SIZE < n) ? tile_start + TILE_SIZE : n;
        store_B_inner: for (int i = tile_start; i < tile_end; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=TILE_SIZE
#pragma HLS DEPENDENCE variable=lB inter false
            B[i] = lB[i];
        }
    }
}

} // extern "C"