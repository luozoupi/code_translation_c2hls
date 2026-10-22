#include "kernel.h"

// ---------------------------------------------------------------------------
// Tiled, locally-staged SGEMM:  C[i][j] = sum_k A[i][k] * B[j][k]
//   * A (I x K) and B (J x K) are both row-major with K contiguous.
//   * Full-K operand tiles are kept on chip so every B (and A) element that is
//     reused is only fetched once per reuse window.
//   * 32x32 output tile  ->  1024 parallel fp32 MACs (≈5 DSP each => ≈5120 DSP,
//     well below the 9024 device cap).
//   * 512-bit (LANES = 16 floats) burst loads/stores on three distinct bundles.
//   * Every load / compute / store loop is pipelined; all inner work is
//     fully unrolled and the local buffers are partitioned to match.
// ---------------------------------------------------------------------------

#define TI    32                 // output tile rows  (A rows)
#define TJ    32                 // output tile cols  (B rows)
#define LANES 16                 // 512-bit / 32-bit  = 16 floats per AXI beat

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {

#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 max_widen_bitwidth=512 max_read_burst_length=64  num_read_outstanding=16  latency=64
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 max_widen_bitwidth=512 max_read_burst_length=64  num_read_outstanding=16  latency=64
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 max_widen_bitwidth=512 max_write_burst_length=64 num_write_outstanding=16 latency=64
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    data_t A_tile[TI][K];        // 32 x 1024  (resident across the whole j0 sweep)
    data_t B_tile[TJ][K];        // 32 x 1024  (resident for one (i0,j0) tile)
    data_t C_tile[TI][TJ];       // 32 x 32 accumulator registers

    // A_tile: bank rows by TI so all 32 rows can be read at a fixed k,
    //         bank the k dimension by LANES so the wide load has 16 write ports.
#pragma HLS ARRAY_PARTITION variable=A_tile cyclic factor=TI    dim=1
#pragma HLS ARRAY_PARTITION variable=A_tile cyclic factor=LANES dim=2
    // B_tile: same banking scheme.
#pragma HLS ARRAY_PARTITION variable=B_tile cyclic factor=TJ    dim=1
#pragma HLS ARRAY_PARTITION variable=B_tile cyclic factor=LANES dim=2
    // Accumulators live in registers.
#pragma HLS ARRAY_PARTITION variable=C_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=C_tile complete dim=2

    // -----------------------------------------------------------------
    // Outer sweep over row blocks of A / C
    // -----------------------------------------------------------------
    tile_i: for (int i0 = 0; i0 < I; i0 += TI) {

        // ---- bulk stage of the A row block (reused by every j0) ----
        load_A: for (int ii = 0; ii < TI; ii++) {
            load_A_k: for (int kb = 0; kb < K; kb += LANES) {
#pragma HLS PIPELINE II=1
                load_A_l: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
                    A_tile[ii][kb + l] = A[i0 + ii][kb + l];
                }
            }
        }

        // -----------------------------------------------------------------
        // Inner sweep over column blocks of B / C
        // -----------------------------------------------------------------
        tile_j: for (int j0 = 0; j0 < J; j0 += TJ) {

            // ---- bulk stage of the B row block ----
            load_B: for (int jj = 0; jj < TJ; jj++) {
                load_B_k: for (int kb = 0; kb < K; kb += LANES) {
#pragma HLS PIPELINE II=1
                    load_B_l: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
                        B_tile[jj][kb + l] = B[j0 + jj][kb + l];
                    }
                }
            }

            // ---- clear the local accumulator tile ----
            init_C: for (int ii = 0; ii < TI; ii++) {
#pragma HLS UNROLL
                init_C_j: for (int jj = 0; jj < TJ; jj++) {
#pragma HLS UNROLL
                    C_tile[ii][jj] = (data_t)0;
                }
            }

            // ---- main compute: 1024 parallel MACs on local memory ----
            compute_k: for (int k = 0; k < K; k++) {
#pragma HLS PIPELINE II=1
                compute_i: for (int ii = 0; ii < TI; ii++) {
#pragma HLS UNROLL
                    compute_j: for (int jj = 0; jj < TJ; jj++) {
#pragma HLS UNROLL
                        C_tile[ii][jj] += A_tile[ii][k] * B_tile[jj][k];
                    }
                }
            }

            // ---- bulk store of the finished tile ----
            store_C: for (int ii = 0; ii < TI; ii++) {
                store_C_j: for (int jb = 0; jb < TJ; jb += LANES) {
#pragma HLS PIPELINE II=1
                    store_C_l: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
                        C[i0 + ii][j0 + jb + l] = C_tile[ii][jb + l];
                    }
                }
            }
        }
    }
}