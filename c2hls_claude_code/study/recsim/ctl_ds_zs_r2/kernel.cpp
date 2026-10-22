#include "kernel.h"

// ---------------------------------------------------------------------------
//  Blocking factors (tile sizes).  I = J = K = 1024 are exact multiples of
//  these values, so no remainder handling is required.
//    TI : rows of A / rows of C  handled per tile
//    TJ : rows of B (= columns of C) handled per tile
//    TK : depth of the k-tile  (= full unroll factor of the MAC loop)
// ---------------------------------------------------------------------------
#define TI 32
#define TJ 64
#define TK 16

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {

    // -----------------------------------------------------------------------
    // Top level interfaces: three independent AXI4 masters (A, B, C) plus an
    // AXI4-Lite control interface.  The matrices are far too large for BRAM,
    // so they must live in off-chip memory.
    // -----------------------------------------------------------------------
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=I*K max_read_burst_length=64  num_read_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=J*K max_read_burst_length=64  num_read_outstanding=16
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=I*J max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // -----------------------------------------------------------------------
    // On-chip tiles.  Partitioned so that the fully unrolled TK-wide MAC loop
    // can read TK operands of A_tile / B_tile in a single clock cycle.
    // -----------------------------------------------------------------------
    data_t A_tile[TI][TK];
    data_t B_tile[TJ][TK];
    data_t C_tile[TI][TJ];
#pragma HLS ARRAY_PARTITION variable=A_tile cyclic factor=TK dim=2
#pragma HLS ARRAY_PARTITION variable=B_tile cyclic factor=TK dim=2
#pragma HLS ARRAY_PARTITION variable=C_tile cyclic factor=8  dim=2

    // -----------------------------------------------------------------------
    // Outer blocking: iterate over tiles of the output matrix.
    // -----------------------------------------------------------------------
    for (int i0 = 0; i0 < I; i0 += TI) {
        for (int j0 = 0; j0 < J; j0 += TJ) {

            // ---- clear the accumulator tile -------------------------------
            for (int i = 0; i < TI; i++) {
#pragma HLS PIPELINE II=1
                for (int j = 0; j < TJ; j++) {
                    C_tile[i][j] = (data_t)0;
                }
            }

            // ---- reduction over the k dimension, tile by tile -------------
            for (int k0 = 0; k0 < K; k0 += TK) {

                // Load the A tile (TI x TK).  The inner loop walks contiguous
                // memory, so Vitis HLS widens the port and issues bursts.
                for (int i = 0; i < TI; i++) {
#pragma HLS PIPELINE II=1
                    for (int k = 0; k < TK; k++) {
#pragma HLS UNROLL
                        A_tile[i][k] = A[i0 + i][k0 + k];
                    }
                }

                // Load the B tile (TJ x TK)
                for (int j = 0; j < TJ; j++) {
#pragma HLS PIPELINE II=1
                    for (int k = 0; k < TK; k++) {
#pragma HLS UNROLL
                        B_tile[j][k] = B[j0 + j][k0 + k];
                    }
                }

                // ---- compute ----------------------------------------------
                // (i,j) are flattened into one loop so the pipeline never
                // drains between rows.  The k loop is fully unrolled, which
                // gives TK independent multiplies and lets HLS replicate the
                // adder chain to sustain II = 1 (TK MACs / clock).
                for (int ij = 0; ij < TI * TJ; ij++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=C_tile type=inter dependent=false
                    const int i = ij / TJ;
                    const int j = ij % TJ;

                    data_t acc = C_tile[i][j];
                    for (int k = 0; k < TK; k++) {
#pragma HLS UNROLL
                        acc += A_tile[i][k] * B_tile[j][k];
                    }
                    C_tile[i][j] = acc;
                }
            }

            // ---- write the finished C tile back to off-chip memory --------
            for (int i = 0; i < TI; i++) {
#pragma HLS PIPELINE II=1
                for (int j = 0; j < TJ; j++) {
                    C[i0 + i][j0 + j] = C_tile[i][j];
                }
            }
        }
    }
}