#include "kernel.h"

// ---------------------------------------------------------------------------
// Output-stationary GEMM engine:  C[i][j] += sum_k A[i][k] * B[j][k]
//   per cycle: TI_U x TJ_U PEs, each a KU-wide dot product
//   = 11 x 16 x 16 = 2816 MACs/cycle  (fmul fulldsp = 3 DSP -> 8448 DSP)
//   all fadds bound to fabric (deterministic DSP count)
//   A and B fully resident on chip (URAM, explicitly banked)
//   accumulator recurrence hidden by interleaving S_INT output sub-blocks
//   I padded to I_PAD (multiple of TI); pad rows of A are zero, never stored
// ---------------------------------------------------------------------------
#define TI_U   11                // PE rows   (i)
#define TJ_U   16                // PE cols   (j) == LANES
#define KU     16                // k-unroll per PE (== LANES)
#define LANES  16                // 512-bit beat / 32-bit float
#define SI     8                 // interleaved sub-blocks along i
#define SJ     8                 // interleaved sub-blocks along j
#define S_INT  (SI * SJ)         // 64 interleaved accumulators per PE
#define TI     (TI_U * SI)       // 88  output rows per tile
#define TJ     (TJ_U * SJ)       // 128 output cols per tile
#define NT_I   ((I + TI - 1) / TI)   // 12 row tiles
#define NT_J   (J / TJ)              // 8 col tiles
#define I_PAD  (NT_I * TI)           // 1056
#define KB     (K / KU)              // 64 k-blocks
#define A_GRPS (I_PAD / TI_U)        // 96
#define A_DEPTH (A_GRPS * KB)        // 6144
#define B_DEPTH ((J / TJ_U) * KB)    // 4096
#define CB_PER_ROW (TJ / LANES)      // 8 beats per C tile row

static data_t mul_dsp(data_t a, data_t b) {
#pragma HLS INLINE
  data_t r = a * b;
#pragma HLS BIND_OP variable=r op=fmul impl=fulldsp
  return r;
}

static data_t add_fabric(data_t a, data_t b) {
#pragma HLS INLINE
  data_t r = a + b;
#pragma HLS BIND_OP variable=r op=fadd impl=fabric
  return r;
}

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=16 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  // A_bank[i % TI_U][k % KU][(i / TI_U) * KB + k / KU]
  static data_t A_bank[TI_U][KU][A_DEPTH];
#pragma HLS ARRAY_PARTITION variable=A_bank complete dim=1
#pragma HLS ARRAY_PARTITION variable=A_bank complete dim=2
#pragma HLS BIND_STORAGE variable=A_bank type=ram_2p impl=uram

  // B_bank[j % TJ_U][k % KU][(j / TJ_U) * KB + k / KU]
  static data_t B_bank[TJ_U][KU][B_DEPTH];
#pragma HLS ARRAY_PARTITION variable=B_bank complete dim=1
#pragma HLS ARRAY_PARTITION variable=B_bank complete dim=2
#pragma HLS BIND_STORAGE variable=B_bank type=ram_2p impl=uram

  // acc[ii][jj][si*SJ + sj] <-> C[it*TI + si*TI_U + ii][jt*TJ + sj*TJ_U + jj]
  data_t acc[TI_U][TJ_U][S_INT];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc complete dim=2
#pragma HLS BIND_STORAGE variable=acc type=ram_2p impl=lutram

  // ---------------- load A and B once (parallel bundles, 512-bit beats) ----
  {
    int a_bank = 0, a_grp = 0;   // r % TI_U, r / TI_U
    int b_bank = 0, b_grp = 0;   // r % TJ_U, r / TJ_U
    int r = 0, kb = 0;
    load_ab: for (int t = 0; t < I * KB; t++) {
#pragma HLS PIPELINE II=1
      load_ab_lane: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
        A_bank[a_bank][l][a_grp * KB + kb] = A[r][kb * KU + l];
        B_bank[b_bank][l][b_grp * KB + kb] = B[r][kb * KU + l];
      }
      if (kb == KB - 1) {
        kb = 0;
        r++;
        if (a_bank == TI_U - 1) { a_bank = 0; a_grp++; } else { a_bank++; }
        if (b_bank == TJ_U - 1) { b_bank = 0; b_grp++; } else { b_bank++; }
      } else {
        kb++;
      }
    }

    // zero padded A rows (I .. I_PAD-1); continue the same counters
    pad_a: for (int t = 0; t < (I_PAD - I) * KB; t++) {
#pragma HLS PIPELINE II=1
      pad_a_lane: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
        A_bank[a_bank][l][a_grp * KB + kb] = 0;
      }
      if (kb == KB - 1) {
        kb = 0;
        if (a_bank == TI_U - 1) { a_bank = 0; a_grp++; } else { a_bank++; }
      } else {
        kb++;
      }
    }
  }

  // ---------------- output tiles ----------------
  tile_i: for (int it = 0; it < NT_I; it++) {
    tile_j: for (int jt = 0; jt < NT_J; jt++) {

      // load C tile into accumulators (one 512-bit beat per cycle)
      {
        int rb = 0, rg = 0, r = 0, jb = 0;   // rb = r % TI_U, rg = r / TI_U
        load_c: for (int t = 0; t < TI * CB_PER_ROW; t++) {
#pragma HLS PIPELINE II=1
          int gr = it * TI + r;
          int s  = rg * SJ + jb;
          load_c_lane: for (int jj = 0; jj < LANES; jj++) {
#pragma HLS UNROLL
            data_t v = 0;
            if (gr < I) v = C[gr][jt * TJ + jb * LANES + jj];
            acc[rb][jj][s] = v;
          }
          if (jb == CB_PER_ROW - 1) {
            jb = 0;
            r++;
            if (rb == TI_U - 1) { rb = 0; rg++; } else { rb++; }
          } else {
            jb++;
          }
        }
      }

      // compute: 2816 MACs / cycle, accumulator reuse distance = S_INT (64)
      compute: for (int t = 0; t < KB * S_INT; t++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=acc type=inter false
        int ks = t / S_INT;
        int s  = t % S_INT;
        int si = s / SJ;
        int sj = s % SJ;
        int aoff = (it * SI + si) * KB + ks;
        int boff = (jt * SJ + sj) * KB + ks;

        data_t a_reg[TI_U][KU];
#pragma HLS ARRAY_PARTITION variable=a_reg complete dim=0
        data_t b_reg[TJ_U][KU];
#pragma HLS ARRAY_PARTITION variable=b_reg complete dim=0

        read_a_i: for (int ii = 0; ii < TI_U; ii++) {
#pragma HLS UNROLL
          read_a_k: for (int kk = 0; kk < KU; kk++) {
#pragma HLS UNROLL
            a_reg[ii][kk] = A_bank[ii][kk][aoff];
          }
        }
        read_b_j: for (int jj = 0; jj < TJ_U; jj++) {
#pragma HLS UNROLL
          read_b_k: for (int kk = 0; kk < KU; kk++) {
#pragma HLS UNROLL
            b_reg[jj][kk] = B_bank[jj][kk][boff];
          }
        }

        pe_i: for (int ii = 0; ii < TI_U; ii++) {
#pragma HLS UNROLL
          pe_j: for (int jj = 0; jj < TJ_U; jj++) {
#pragma HLS UNROLL
            data_t p[KU];
#pragma HLS ARRAY_PARTITION variable=p complete
            pe_mul: for (int kk = 0; kk < KU; kk++) {
#pragma HLS UNROLL
              p[kk] = mul_dsp(a_reg[ii][kk], b_reg[jj][kk]);
            }
            data_t s1[8];
#pragma HLS ARRAY_PARTITION variable=s1 complete
            tree_l1: for (int q = 0; q < 8; q++) {
#pragma HLS UNROLL
              s1[q] = add_fabric(p[2 * q], p[2 * q + 1]);
            }
            data_t s2[4];
#pragma HLS ARRAY_PARTITION variable=s2 complete
            tree_l2: for (int q = 0; q < 4; q++) {
#pragma HLS UNROLL
              s2[q] = add_fabric(s1[2 * q], s1[2 * q + 1]);
            }
            data_t s3[2];
#pragma HLS ARRAY_PARTITION variable=s3 complete
            tree_l3: for (int q = 0; q < 2; q++) {
#pragma HLS UNROLL
              s3[q] = add_fabric(s2[2 * q], s2[2 * q + 1]);
            }
            data_t dot = add_fabric(s3[0], s3[1]);
            acc[ii][jj][s] = add_fabric(acc[ii][jj][s], dot);
          }
        }
      }

      // store C tile (one 512-bit beat per cycle), skip padded rows
      {
        int rb = 0, rg = 0, r = 0, jb = 0;
        store_c: for (int t = 0; t < TI * CB_PER_ROW; t++) {
#pragma HLS PIPELINE II=1
          int gr = it * TI + r;
          int s  = rg * SJ + jb;
          if (gr < I) {
            store_c_lane: for (int jj = 0; jj < LANES; jj++) {
#pragma HLS UNROLL
              C[gr][jt * TJ + jb * LANES + jj] = acc[rb][jj][s];
            }
          }
          if (jb == CB_PER_ROW - 1) {
            jb = 0;
            r++;
            if (rb == TI_U - 1) { rb = 0; rg++; } else { rb++; }
          } else {
            jb++;
          }
        }
      }
    }
  }
}