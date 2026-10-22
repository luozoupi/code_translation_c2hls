#include "kernel.h"

/* ------------------------------------------------------------------
 * autosa_mm  :  C[i][j] = sum_k A[i][k] * B[j][k]   (I=J=K=1024)
 *
 * Baseline was a reference nest:  the k-loop read-modify-wrote the
 * m_axi port C[i][j] every iteration, giving II=7 and ~7.7e9 cycles.
 *
 * Flash rewrite:
 *   - classic blocked GEMM with full load / compute / store separation
 *   - A row-panel (TILE_I x K) staged once per row tile and kept
 *     resident on chip across every column tile (A moves ~4 MB total)
 *   - B row-panel (TILE_J x K) staged per (row,column) tile
 *   - TILE_I x TILE_J private register accumulators, k-loop pipelined
 *   - bulk load / store loops walk LANES = 512/32 = 16 floats per beat
 *     with matching consecutive-dimension cyclic banking
 *   - distinct m_axi bundles (gmem0/1/2) so A, B, C do not contend
 * ------------------------------------------------------------------ */

#define TILE_I 64          /* rows of A / C kept on chip            */
#define TILE_J 32          /* columns of C (rows of B) per tile     */
#define LANES  16          /* 512-bit beat / 32-bit element         */

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {

#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 max_widen_bitwidth=512 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    /* ---------------- on-chip working set ------------------------- */
    data_t A_blk[TILE_I][K];      /* 64 x 1024 floats = 256 KB        */
    data_t B_blk[TILE_J][K];      /* 32 x 1024 floats = 128 KB        */
    data_t acc[TILE_I][TILE_J];   /* 2048 private accumulators        */

    /* A_blk[i][k] : bank = (i, k%LANES)  -> TILE_I*LANES banks
       gives one read per bank per cycle for the k-column access and
       LANES independent write ports for the 512-bit load walk.       */
#pragma HLS ARRAY_PARTITION variable=A_blk complete dim=1
#pragma HLS ARRAY_PARTITION variable=A_blk cyclic factor=LANES dim=2
#pragma HLS ARRAY_PARTITION variable=B_blk complete dim=1
#pragma HLS ARRAY_PARTITION variable=B_blk cyclic factor=LANES dim=2
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0

    /* ================= row tile (A panel resident) ================= */
    row_tile: for (int i0 = 0; i0 < I; i0 += TILE_I) {

        /* ---- bulk load A panel, 512-bit contiguous beats ---------- */
        load_A: for (int i = 0; i < TILE_I; ++i) {
            load_A_k: for (int k0 = 0; k0 < K; k0 += LANES) {
#pragma HLS PIPELINE II=1
                load_A_l: for (int l = 0; l < LANES; ++l) {
#pragma HLS UNROLL
                    A_blk[i][k0 + l] = A[i0 + i][k0 + l];
                }
            }
        }

        /* ================= column tile ============================= */
        col_tile: for (int j0 = 0; j0 < J; j0 += TILE_J) {

            /* ---- bulk load B panel, 512-bit contiguous beats ------ */
            load_B: for (int j = 0; j < TILE_J; ++j) {
                load_B_k: for (int k0 = 0; k0 < K; k0 += LANES) {
#pragma HLS PIPELINE II=1
                    load_B_l: for (int l = 0; l < LANES; ++l) {
#pragma HLS UNROLL
                        B_blk[j][k0 + l] = B[j0 + j][k0 + l];
                    }
                }
            }

            /* ---- clear private accumulators (registers) ----------- */
            clear_acc_i: for (int i = 0; i < TILE_I; ++i) {
#pragma HLS UNROLL
                clear_acc_j: for (int j = 0; j < TILE_J; ++j) {
#pragma HLS UNROLL
                    acc[i][j] = (data_t)0;
                }
            }

            /* ---- local-memory reduction, pipelined k --------------
             * Every operand comes from on-chip memory; the only
             * loop-carried dependence is the FP accumulate chain,
             * which sets the (legal) II of this pipeline.          */
            compute_k: for (int k = 0; k < K; ++k) {
#pragma HLS PIPELINE II=1
                compute_i: for (int i = 0; i < TILE_I; ++i) {
#pragma HLS UNROLL
                    compute_j: for (int j = 0; j < TILE_J; ++j) {
#pragma HLS UNROLL
                        acc[i][j] += A_blk[i][k] * B_blk[j][k];
                    }
                }
            }

            /* ---- bulk store C tile, 512-bit contiguous beats ------ */
            store_C: for (int i = 0; i < TILE_I; ++i) {
                store_C_j: for (int jj = 0; jj < TILE_J; jj += LANES) {
#pragma HLS PIPELINE II=1
                    store_C_l: for (int l = 0; l < LANES; ++l) {
#pragma HLS UNROLL
                        C[i0 + i][j0 + jj + l] = acc[i][jj + l];
                    }
                }
            }
        }
    }
}