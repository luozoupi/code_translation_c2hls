#include "kernel.h"

// Processing-array geometry (27 x 64 = 1728 parallel FP MACs, ~8640 DSP)
#define PI     27                    // parallel rows of A (i) per block
#define PJ     64                    // parallel rows of B (j) per cycle
#define NJ     (J / PJ)              // 16 interleaved j-groups (hides FP add latency)
#define LANES  16                    // 512-bit beat / 32-bit float
#define NIB    ((I + PI - 1) / PI)   // 38 i-blocks (last one partially valid)
#define KC     (K / LANES)           // 64 beats per A/B row
#define JC     (J / LANES)           // 64 beats per C row
#define QC     (PJ / LANES)          // 4 beats per j-group

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  // Full B resident on chip: 64 banks on j (compute width) x 16 banks on k (AXI beat)
  data_t Bloc[J][K];
#pragma HLS ARRAY_PARTITION variable=Bloc cyclic factor=64 dim=1
#pragma HLS ARRAY_PARTITION variable=Bloc cyclic factor=16 dim=2
#pragma HLS BIND_STORAGE variable=Bloc type=ram_2p impl=bram

  // A rows of the current i-block
  data_t Aloc[PI][K];
#pragma HLS ARRAY_PARTITION variable=Aloc complete dim=1
#pragma HLS ARRAY_PARTITION variable=Aloc cyclic factor=16 dim=2

  // Output-stationary accumulators: acc[p][q][jj] holds C[i0+p][jj*PJ+q]
  data_t acc[PI][PJ][NJ];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc complete dim=2

  // ---------------- Load all of B once (16 floats / cycle) ----------------
  load_B_row: for (int j = 0; j < J; j++) {
    load_B_beat: for (int kc = 0; kc < KC; kc++) {
#pragma HLS PIPELINE II=1
      load_B_lane: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
        Bloc[j][kc * LANES + l] = B[j][kc * LANES + l];
      }
    }
  }

  // ---------------- Process i-blocks ----------------
  block_i: for (int ib = 0; ib < NIB; ib++) {
    const int i0 = ib * PI;

    // Load A rows and C rows (seed accumulators) concurrently: gmem0 + gmem2
    load_AC_row: for (int p = 0; p < PI; p++) {
      const int  i     = i0 + p;
      const bool valid = (i < I);
      const int  ir    = valid ? i : (I - 1);   // clamp: always-legal burst read
      load_AC_beat: for (int c = 0; c < KC; c++) {
#pragma HLS PIPELINE II=1
        const int jj = c / QC;
        const int qb = (c % QC) * LANES;
        load_AC_lane: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
          data_t av = A[ir][c * LANES + l];
          data_t cv = (data_t)0;
          Aloc[p][c * LANES + l] = valid ? av : (data_t)0;
          acc[p][qb + l][jj]     = valid ? cv : (data_t)0;
        }
      }
    }

    // Compute: PI x PJ MACs per cycle; acc reuse distance = NJ iterations
    compute_k: for (int k = 0; k < K; k++) {
      compute_jj: for (int jj = 0; jj < NJ; jj++) {
#pragma HLS PIPELINE II=1
        pe_i: for (int p = 0; p < PI; p++) {
#pragma HLS UNROLL
          data_t a = Aloc[p][k];
          pe_j: for (int q = 0; q < PJ; q++) {
#pragma HLS UNROLL
            acc[p][q][jj] = acc[p][q][jj] + a * Bloc[jj * PJ + q][k];
          }
        }
      }
    }

    // Store finished C rows once
    store_C_row: for (int p = 0; p < PI; p++) {
      const int i = i0 + p;
      if (i < I) {
        store_C_beat: for (int c = 0; c < JC; c++) {
#pragma HLS PIPELINE II=1
          const int jj = c / QC;
          const int qb = (c % QC) * LANES;
          store_C_lane: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
            C[i][c * LANES + l] = acc[p][qb + l][jj];
          }
        }
      }
    }
  }
}