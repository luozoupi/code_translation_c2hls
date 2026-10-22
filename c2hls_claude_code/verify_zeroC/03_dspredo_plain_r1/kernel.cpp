#include "kernel.h"

// Spatial PE array: T_I rows of A  x  T_J rows of B  x  T_K reduction lanes
// 7 x 16 x 16 = 1792 MACs/cycle  (~5 DSP per float MAC -> ~8960 DSP < 9024)
#define T_I 7
#define T_J 16
#define T_K 16
#define LANES 16                       // 512-bit beat / 32-bit float
#define NT ((I + T_I - 1) / T_I)       // row tiles (last one partial, guarded)
#define KB (K / T_K)                   // k chunks
#define JB (J / T_J)                   // j blocks
#define KBL (K / LANES)                // load beats per A/B row
#define JBL (J / LANES)                // load/store beats per C row

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  // Full B on-chip: 16 x 16 banks of 64x64 floats (1 URAM each)
  static data_t B_loc[J][K];
#pragma HLS ARRAY_PARTITION variable=B_loc cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=B_loc cyclic factor=16 dim=2
#pragma HLS BIND_STORAGE variable=B_loc type=ram_1p impl=uram

  // A row tile: 7 x 16 banks
  static data_t A_loc[T_I][K];
#pragma HLS ARRAY_PARTITION variable=A_loc complete dim=1
#pragma HLS ARRAY_PARTITION variable=A_loc cyclic factor=16 dim=2

  // C row tile accumulators: 7 x 16 banks
  static data_t acc[T_I][J];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc cyclic factor=16 dim=2

  // ---------------- load B once (16 floats / beat) ----------------
  load_B: for (int idx = 0; idx < J * KBL; ++idx) {
#pragma HLS PIPELINE II=1
    int j  = idx / KBL;
    int kb = (idx % KBL) * LANES;
    load_B_lane: for (int l = 0; l < LANES; ++l) {
#pragma HLS UNROLL
      B_loc[j][kb + l] = B[j][kb + l];
    }
  }

  // ---------------- row tiles of A / C ----------------
  tile_i: for (int t = 0; t < NT; ++t) {
    int i0 = t * T_I;

    // load A rows and C rows (separate bundles -> same cycle)
    load_AC: for (int idx = 0; idx < T_I * KBL; ++idx) {
#pragma HLS PIPELINE II=1
      int r  = idx / KBL;
      int kb = (idx % KBL) * LANES;
      int gi = i0 + r;
      if (gi > I - 1) gi = I - 1;   // clamp: padded rows are never stored
      load_AC_lane: for (int l = 0; l < LANES; ++l) {
#pragma HLS UNROLL
        A_loc[r][kb + l] = A[gi][kb + l];
        acc[r][kb + l]   = (data_t)0;   // J == K == 1024
      }
    }

    // compute: 1792 MACs / cycle, acc address reused every JB=64 iterations
    compute_k: for (int kc = 0; kc < KB; ++kc) {
      compute_j: for (int jb = 0; jb < JB; ++jb) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=acc inter RAW distance=64 true
        pe_i: for (int pi = 0; pi < T_I; ++pi) {
#pragma HLS UNROLL
          pe_j: for (int pj = 0; pj < T_J; ++pj) {
#pragma HLS UNROLL
            data_t p[T_K];
#pragma HLS ARRAY_PARTITION variable=p complete
            data_t s8[8];
#pragma HLS ARRAY_PARTITION variable=s8 complete
            data_t s4[4];
#pragma HLS ARRAY_PARTITION variable=s4 complete
            data_t s2[2];
#pragma HLS ARRAY_PARTITION variable=s2 complete
            mul_k: for (int u = 0; u < T_K; ++u) {
#pragma HLS UNROLL
              p[u] = A_loc[pi][kc * T_K + u] * B_loc[jb * T_J + pj][kc * T_K + u];
            }
            tree_8: for (int u = 0; u < 8; ++u) {
#pragma HLS UNROLL
              s8[u] = p[2 * u] + p[2 * u + 1];
            }
            tree_4: for (int u = 0; u < 4; ++u) {
#pragma HLS UNROLL
              s4[u] = s8[2 * u] + s8[2 * u + 1];
            }
            tree_2: for (int u = 0; u < 2; ++u) {
#pragma HLS UNROLL
              s2[u] = s4[2 * u] + s4[2 * u + 1];
            }
            data_t s = s2[0] + s2[1];
            acc[pi][jb * T_J + pj] = acc[pi][jb * T_J + pj] + s;
          }
        }
      }
    }

    // store valid C rows once
    store_C: for (int idx = 0; idx < T_I * JBL; ++idx) {
#pragma HLS PIPELINE II=1
      int r  = idx / JBL;
      int jb = (idx % JBL) * LANES;
      if (i0 + r < I) {
        store_C_lane: for (int l = 0; l < LANES; ++l) {
#pragma HLS UNROLL
          C[i0 + r][jb + l] = acc[r][jb + l];
        }
      }
    }
  }
}