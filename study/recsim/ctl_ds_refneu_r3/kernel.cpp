#include "kernel.h"

/* ------------------------------------------------------------------ */
/*  C[i][j] = sum_k A[i][k] * B[j][k]        I = J = K = 1024          */
/*                                                                     */
/*  Op-point / tile = 256 x 256 x 128                                  */
/*    global traffic per tile : (256+256)*128/16 = 4096 512-bit beats  */
/*    compute per tile        : 65536 flattened (i,j) iterations       */
/*                              x 128 unrolled MACs, II = 1            */
/*    -> compute : transfer   ~ 16 : 1 (datapath never waits on AXI)   */
/*                                                                     */
/*  A_loc / B_loc are re-used by all 65536 MAC iterations of a chunk,  */
/*  C_loc is accumulated entirely on chip and written back once with a */
/*  512-bit / II=1 store pass.                                         */
/* ------------------------------------------------------------------ */

#define TI      256              /* output tile rows                 */
#define TJ      256              /* output tile cols                 */
#define TK      128              /* k chunk (kept on chip)           */
#define LANES   16               /* 512 bit / 32 bit float           */
#define NPART   16               /* independent partial accumulators */

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {

#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  tile_i: for (int i0 = 0; i0 < I; i0 += TI) {
    tile_j: for (int j0 = 0; j0 < J; j0 += TJ) {

      /* ---------------- on-chip accumulator tile ---------------- */
      data_t C_loc[TI][TJ];
#pragma HLS ARRAY_PARTITION variable=C_loc cyclic factor=LANES dim=2

      init_c: for (int i = 0; i < TI; i++) {
        init_c_l: for (int jj = 0; jj < TJ; jj += LANES) {
#pragma HLS PIPELINE II=1
          init_c_u: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
            C_loc[i][jj + l] = (data_t)0;
          }
        }
      }

      /* ---------------- reduction over K in chunks -------------- */
      chunk_k: for (int k0 = 0; k0 < K; k0 += TK) {

        /* operand tiles stay resident for the whole 256x256 MAC sweep */
        data_t A_loc[TI][TK];
        data_t B_loc[TJ][TK];
#pragma HLS ARRAY_PARTITION variable=A_loc complete dim=2
#pragma HLS ARRAY_PARTITION variable=B_loc complete dim=2

        /* ---- 512-bit contiguous burst load of A tile (gmem0) ---- */
        load_a: for (int i = 0; i < TI; i++) {
          load_a_k: for (int kk = 0; kk < TK; kk += LANES) {
#pragma HLS PIPELINE II=1
            load_a_u: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
              A_loc[i][kk + l] = A[i0 + i][k0 + kk + l];
            }
          }
        }

        /* ---- 512-bit contiguous burst load of B tile (gmem1) ---- */
        load_b: for (int j = 0; j < TJ; j++) {
          load_b_k: for (int kk = 0; kk < TK; kk += LANES) {
#pragma HLS PIPELINE II=1
            load_b_u: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
              B_loc[j][kk + l] = B[j0 + j][k0 + kk + l];
            }
          }
        }

        /* ---- local-memory MAC sweep, flattened (i,j), k unrolled -- */
        compute_i: for (int i = 0; i < TI; i++) {
          compute_j: for (int j = 0; j < TJ; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN

            data_t part[NPART];
#pragma HLS ARRAY_PARTITION variable=part complete dim=1

            init_p: for (int p = 0; p < NPART; p++) {
#pragma HLS UNROLL
              part[p] = (data_t)0;
            }

            mac_g: for (int kg = 0; kg < TK; kg += NPART) {
#pragma HLS UNROLL
              mac_l: for (int l = 0; l < NPART; l++) {
#pragma HLS UNROLL
                part[l] += A_loc[i][kg + l] * B_loc[j][kg + l];
              }
            }

            /* balanced binary reduction of the partials */
            data_t r0 = part[0]  + part[1];
            data_t r1 = part[2]  + part[3];
            data_t r2 = part[4]  + part[5];
            data_t r3 = part[6]  + part[7];
            data_t r4 = part[8]  + part[9];
            data_t r5 = part[10] + part[11];
            data_t r6 = part[12] + part[13];
            data_t r7 = part[14] + part[15];

            data_t s0 = r0 + r1;
            data_t s1 = r2 + r3;
            data_t s2 = r4 + r5;
            data_t s3 = r6 + r7;

            data_t t0 = s0 + s1;
            data_t t1 = s2 + s3;

            C_loc[i][j] = C_loc[i][j] + (t0 + t1);
          }
        }
      }

      /* ---------------- 512-bit store of the finished tile ------ */
      store_c: for (int i = 0; i < TI; i++) {
        store_c_l: for (int jj = 0; jj < TJ; jj += LANES) {
#pragma HLS PIPELINE II=1
          store_c_u: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
            C[i0 + i][j0 + jj + l] = C_loc[i][jj + l];
          }
        }
      }
    }
  }
}