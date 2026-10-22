#include "kernel.h"

// Spatial / temporal tiling parameters
#define TI   27                    // output rows per i-tile (spatial, unrolled)
#define LJ   16                    // output cols per pipeline iteration (spatial, unrolled)
#define KQ   4                     // reduction lanes per pipeline iteration (unrolled + adder tree)
#define NIT  ((I + TI - 1) / TI)   // 38 i-tiles (last one partial, guarded)
#define NJO  (J / LJ)              // 64 column groups -> accumulator reuse distance
#define NKK  (K / KQ)              // 256 reduction steps
#define CLN  16                    // lanes for C load/store (512-bit beat of floats)

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=16 num_write_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=16 num_write_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=16 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  // Full B on-chip (URAM), banked: 16 on j (spatial cols), 4 on k (reduction lanes)
  static data_t Bl[J][K];
#pragma HLS BIND_STORAGE variable=Bl type=ram_2p impl=uram
#pragma HLS ARRAY_PARTITION variable=Bl cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=Bl cyclic factor=4 dim=2

  // A row-block for current i-tile: every row its own bank, 4 k-lanes
  static data_t Al[TI][K];
#pragma HLS ARRAY_PARTITION variable=Al complete dim=1
#pragma HLS ARRAY_PARTITION variable=Al cyclic factor=4 dim=2

  // C row-block accumulator for current i-tile: 27 rows x 16 col banks
  static data_t Cl[TI][J];
#pragma HLS ARRAY_PARTITION variable=Cl complete dim=1
#pragma HLS ARRAY_PARTITION variable=Cl cyclic factor=16 dim=2

  // ---- Load all of B once ----
  load_B_row: for (int j = 0; j < J; j++) {
    load_B_k: for (int k = 0; k < K; k += KQ) {
#pragma HLS PIPELINE II=1
      load_B_lane: for (int l = 0; l < KQ; l++) {
#pragma HLS UNROLL
        Bl[j][k + l] = B[j][k + l];
      }
    }
  }

  // ---- Process i-tiles ----
  tile_i: for (int t = 0; t < NIT; t++) {
    const int i0 = t * TI;

    // Load A rows of this tile (clamp out-of-range rows; results discarded)
    load_A_row: for (int r = 0; r < TI; r++) {
      int gi = i0 + r;
      int gr = (gi < I) ? gi : (I - 1);
      load_A_k: for (int k = 0; k < K; k += KQ) {
#pragma HLS PIPELINE II=1
        load_A_lane: for (int l = 0; l < KQ; l++) {
#pragma HLS UNROLL
          Al[r][k + l] = A[gr][k + l];
        }
      }
    }

    // Load C rows of this tile (initial accumulator values)
    load_C_row: for (int r = 0; r < TI; r++) {
      int gi = i0 + r;
      int gr = (gi < I) ? gi : (I - 1);
      load_C_j: for (int c = 0; c < J; c += CLN) {
#pragma HLS PIPELINE II=1
        load_C_lane: for (int l = 0; l < CLN; l++) {
#pragma HLS UNROLL
          Cl[r][c + l] = C[gr][c + l];
        }
      }
    }

    // Compute: 27 x 16 x 4 = 1728 MACs per cycle, accumulators reused every 64 iters
    compute_k: for (int kk = 0; kk < NKK; kk++) {
      compute_j: for (int jo = 0; jo < NJO; jo++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=Cl inter distance=64 true
        pe_i: for (int ii = 0; ii < TI; ii++) {
#pragma HLS UNROLL
          pe_j: for (int jl = 0; jl < LJ; jl++) {
#pragma HLS UNROLL
            const int j = jo * LJ + jl;
            const int kb = kk * KQ;
            data_t p0 = Al[ii][kb + 0] * Bl[j][kb + 0];
            data_t p1 = Al[ii][kb + 1] * Bl[j][kb + 1];
            data_t p2 = Al[ii][kb + 2] * Bl[j][kb + 2];
            data_t p3 = Al[ii][kb + 3] * Bl[j][kb + 3];
            data_t s01 = p0 + p1;
            data_t s23 = p2 + p3;
            data_t s = s01 + s23;
            Cl[ii][j] = Cl[ii][j] + s;
          }
        }
      }
    }

    // Store C rows of this tile (only valid rows)
    store_C_row: for (int r = 0; r < TI; r++) {
      int gi = i0 + r;
      if (gi < I) {
        store_C_j: for (int c = 0; c < J; c += CLN) {
#pragma HLS PIPELINE II=1
          store_C_lane: for (int l = 0; l < CLN; l++) {
#pragma HLS UNROLL
            C[gi][c + l] = Cl[r][c + l];
          }
        }
      }
    }
  }
}