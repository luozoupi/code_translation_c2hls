#include "kernel.h"

// -----------------------------------------------------------------------------
// Tiling parameters
// -----------------------------------------------------------------------------
#define TILE_I 16
#define TILE_J 16
#define TILE_K 16

// -----------------------------------------------------------------------------
// Top function: C = A * B^T   (A: I x K, B: J x K, C: I x J)
//
// Plain triple loop is fully serial in software.  We restructure it into a
// blocked (tiled) form so that the working set of each innermost computation
// fits in on-chip memory, then pipeline/unroll the inner loops so that the
// MAC array is fed a new operand every clock cycle.
// -----------------------------------------------------------------------------
extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {

#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=1048576 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=1048576 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=1048576 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // On-chip tile buffers --------------------------------------------------
    data_t A_tile[TILE_I][TILE_K];
    data_t B_tile[TILE_J][TILE_K];
    data_t C_acc [TILE_I][TILE_J];

    // A_tile / B_tile: partition along the k dimension so the fully unrolled
    // inner k loop can issue TILE_K parallel accesses per clock.
#pragma HLS ARRAY_PARTITION variable=A_tile complete dim=2
#pragma HLS ARRAY_PARTITION variable=B_tile complete dim=2
    // C_acc: fully partitioned so that pipelined (i,j) iterations never
    // contend for the same memory port.
#pragma HLS ARRAY_PARTITION variable=C_acc complete dim=0

    // ------------------ outer tiling over rows of C ------------------------
    for (int i0 = 0; i0 < I; i0 += TILE_I) {

        // ------------------ tiling over columns of C -----------------------
        for (int j0 = 0; j0 < J; j0 += TILE_J) {

            // ---------------------------------------------------------------
            // Initialise the per-tile accumulator
            // ---------------------------------------------------------------
            for (int i = 0; i < TILE_I; i++) {
                for (int j = 0; j < TILE_J; j++) {
#pragma HLS PIPELINE II=1
                    C_acc[i][j] = (data_t)0.0f;
                }
            }

            // ---------------------------------------------------------------
            // Reduction over the K dimension in blocks of TILE_K
            // ---------------------------------------------------------------
            for (int k0 = 0; k0 < K; k0 += TILE_K) {

                // ---- load A tile  (TILE_I x TILE_K) -----------------------
                for (int i = 0; i < TILE_I; i++) {
#pragma HLS PIPELINE II=1
                    for (int k = 0; k < TILE_K; k++) {
#pragma HLS UNROLL
                        A_tile[i][k] = A[i0 + i][k0 + k];
                    }
                }

                // ---- load B tile  (TILE_J x TILE_K) -----------------------
                for (int j = 0; j < TILE_J; j++) {
#pragma HLS PIPELINE II=1
                    for (int k = 0; k < TILE_K; k++) {
#pragma HLS UNROLL
                        B_tile[j][k] = B[j0 + j][k0 + k];
                    }
                }

                // ---- MAC over the tile ------------------------------------
                // The (i,j) loop is pipelined with II=1; the fully unrolled k
                // loop gives TILE_K fused multiply-adds per clock cycle.
                for (int i = 0; i < TILE_I; i++) {
                    for (int j = 0; j < TILE_J; j++) {
#pragma HLS PIPELINE II=1
                        data_t acc = C_acc[i][j];
                        for (int k = 0; k < TILE_K; k++) {
#pragma HLS UNROLL
                            acc += A_tile[i][k] * B_tile[j][k];
                        }
                        C_acc[i][j] = acc;
                    }
                }
            }

            // ---------------------------------------------------------------
            // Write the finished C tile back to global memory
            // ---------------------------------------------------------------
            for (int i = 0; i < TILE_I; i++) {
#pragma HLS PIPELINE II=1
                for (int j = 0; j < TILE_J; j++) {
#pragma HLS UNROLL
                    C[i0 + i][j0 + j] = C_acc[i][j];
                }
            }
        }
    }
}