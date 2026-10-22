#include "kernel.h"

// C[i][j] = sum_k A[i][k] * B[j][k]
// A is staged fully on-chip (URAM, banked by UK on k).
// For each block of PJ rows of B (staged on-chip), all 1024 rows of A are
// streamed through a PJ x UK MAC array.  NI rows of A are interleaved in the
// pipeline so that each accumulator is updated only every NI cycles.

#define UK 16              // k lanes per PE (adder tree depth 4)
#define PJ 128             // parallel output columns (PEs)
#define NI 16              // interleaved rows (hides fadd latency)
#define KC (K / UK)        // k chunks
#define NJB (J / PJ)       // j blocks
#define NIG (I / NI)       // i groups

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 max_widen_bitwidth=512 max_read_burst_length=256 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 max_widen_bitwidth=512 max_read_burst_length=256 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 max_widen_bitwidth=512 max_write_burst_length=256 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  static data_t A_loc[I][K];
#pragma HLS ARRAY_PARTITION variable=A_loc cyclic factor=UK dim=2
#pragma HLS BIND_STORAGE variable=A_loc type=ram_2p impl=uram

  static data_t B_blk[PJ][K];
#pragma HLS ARRAY_PARTITION variable=B_blk complete dim=1
#pragma HLS ARRAY_PARTITION variable=B_blk cyclic factor=UK dim=2
#pragma HLS BIND_STORAGE variable=B_blk type=ram_2p impl=lutram

  static data_t C_blk[I][PJ];
#pragma HLS ARRAY_PARTITION variable=C_blk complete dim=2

  static data_t acc[NI][PJ];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=2

load_A_row:
  for (int i = 0; i < I; i++) {
  load_A_chunk:
    for (int kc = 0; kc < KC; kc++) {
#pragma HLS PIPELINE II=1
    load_A_lane:
      for (int u = 0; u < UK; u++) {
#pragma HLS UNROLL
        A_loc[i][kc * UK + u] = A[i][kc * UK + u];
      }
    }
  }

j_block:
  for (int jb = 0; jb < NJB; jb++) {
    const int j0 = jb * PJ;

  load_B_row:
    for (int jj = 0; jj < PJ; jj++) {
    load_B_chunk:
      for (int kc = 0; kc < KC; kc++) {
#pragma HLS PIPELINE II=1
      load_B_lane:
        for (int u = 0; u < UK; u++) {
#pragma HLS UNROLL
          B_blk[jj][kc * UK + u] = B[j0 + jj][kc * UK + u];
        }
      }
    }

  compute_ig:
    for (int ig = 0; ig < NIG; ig++) {
    compute_kc:
      for (int kc = 0; kc < KC; kc++) {
      compute_ii:
        for (int ii = 0; ii < NI; ii++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=acc inter false
#pragma HLS DEPENDENCE variable=C_blk inter false
          const int i = ig * NI + ii;
          data_t a[UK];
#pragma HLS ARRAY_PARTITION variable=a complete
        read_a:
          for (int u = 0; u < UK; u++) {
#pragma HLS UNROLL
            a[u] = A_loc[i][kc * UK + u];
          }
        pe_col:
          for (int jj = 0; jj < PJ; jj++) {
#pragma HLS UNROLL
            data_t p[UK];
#pragma HLS ARRAY_PARTITION variable=p complete
          pe_mul:
            for (int u = 0; u < UK; u++) {
#pragma HLS UNROLL
              p[u] = a[u] * B_blk[jj][kc * UK + u];
            }
            data_t s1[8];
#pragma HLS ARRAY_PARTITION variable=s1 complete
          tree_l1:
            for (int u = 0; u < 8; u++) {
#pragma HLS UNROLL
              s1[u] = p[2 * u] + p[2 * u + 1];
            }
            data_t s2[4];
#pragma HLS ARRAY_PARTITION variable=s2 complete
#pragma HLS BIND_OP variable=s2 op=fadd impl=fabric
          tree_l2:
            for (int u = 0; u < 4; u++) {
#pragma HLS UNROLL
              s2[u] = s1[2 * u] + s1[2 * u + 1];
            }
            data_t s3[2];
#pragma HLS ARRAY_PARTITION variable=s3 complete
#pragma HLS BIND_OP variable=s3 op=fadd impl=fabric
          tree_l3:
            for (int u = 0; u < 2; u++) {
#pragma HLS UNROLL
              s3[u] = s2[2 * u] + s2[2 * u + 1];
            }
            data_t s4 = s3[0] + s3[1];
            data_t prev = (kc == 0) ? (data_t)0 : acc[ii][jj];
            data_t sum = prev + s4;
            acc[ii][jj] = sum;
            if (kc == KC - 1) {
              C_blk[i][jj] = sum;
            }
          }
        }
      }
    }

  store_C_row:
    for (int i = 0; i < I; i++) {
    store_C_chunk:
      for (int jv = 0; jv < PJ / UK; jv++) {
#pragma HLS PIPELINE II=1
      store_C_lane:
        for (int u = 0; u < UK; u++) {
#pragma HLS UNROLL
          C[i][j0 + jv * UK + u] = C_blk[i][jv * UK + u];
        }
      }
    }
  }
}
