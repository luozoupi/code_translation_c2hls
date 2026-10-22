#include "kernel.h"

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
#define LANES      16                      // 512-bit beat = 16 floats
#define TI         104                     // rows of A per tile (spatial PEs)
#define NT         ((I + TI - 1) / TI)     // number of row tiles (10)
#define KB         (K / LANES)             // beats per A/B row (64)
#define JB         (J / LANES)             // beats per C row (64)
#define JG         LANES                   // B rows per block (16)
#define NBLK       (J / JG)                // B blocks (64)
#define BLK_BEATS  (JG * KB)               // beats per B block (1024)
#define TILE_BEATS (TI * KB)               // A beats per tile

static_assert(KB == JB, "tile loops assume K == J");
static_assert(JB == NBLK, "one C column group per B block");
static_assert((NBLK & (NBLK - 1)) == 0, "NBLK must be a power of two");

// ---------------------------------------------------------------------------
// Balanced 16-input adder tree
// ---------------------------------------------------------------------------
static inline data_t tree16(const data_t p[LANES]) {
#pragma HLS INLINE
  data_t s8[8], s4[4], s2[2];
#pragma HLS ARRAY_PARTITION variable=s8 complete
#pragma HLS ARRAY_PARTITION variable=s4 complete
#pragma HLS ARRAY_PARTITION variable=s2 complete
tree_l8:
  for (int m = 0; m < 8; ++m) {
#pragma HLS UNROLL
    s8[m] = p[2 * m] + p[2 * m + 1];
  }
tree_l4:
  for (int m = 0; m < 4; ++m) {
#pragma HLS UNROLL
    s4[m] = s8[2 * m] + s8[2 * m + 1];
  }
tree_l2:
  for (int m = 0; m < 2; ++m) {
#pragma HLS UNROLL
    s2[m] = s4[2 * m] + s4[2 * m + 1];
  }
  return s2[0] + s2[1];
}

// ---------------------------------------------------------------------------
// Top
// ---------------------------------------------------------------------------
extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  static data_t A_loc[TI][KB][LANES];
#pragma HLS ARRAY_PARTITION variable=A_loc complete dim=1
#pragma HLS ARRAY_PARTITION variable=A_loc complete dim=3
  static data_t C_loc[TI][JB][LANES];
#pragma HLS ARRAY_PARTITION variable=C_loc complete dim=1
#pragma HLS ARRAY_PARTITION variable=C_loc complete dim=3
  static data_t B_buf0[JG][KB][LANES];
#pragma HLS ARRAY_PARTITION variable=B_buf0 complete dim=3
  static data_t B_buf1[JG][KB][LANES];
#pragma HLS ARRAY_PARTITION variable=B_buf1 complete dim=3
  static data_t rowbuf[JB][LANES];
#pragma HLS ARRAY_PARTITION variable=rowbuf complete dim=2

  // prologue: block 0 of B into buffer 0
pro_B:
  for (int q = 0; q < BLK_BEATS; ++q) {
#pragma HLS PIPELINE II=1
    const int r = q / KB;
    const int c = q % KB;
  pro_B_lane:
    for (int l = 0; l < LANES; ++l) {
#pragma HLS UNROLL
      B_buf0[r][c][l] = B[r][c * LANES + l];
    }
  }

tile_loop:
  for (int tt = 0; tt < NT; ++tt) {
    const int base = tt * TI;

    // load A tile (zero-pad rows beyond I), clear accumulators
  load_A_tile:
    for (int q = 0; q < TILE_BEATS; ++q) {
#pragma HLS PIPELINE II=1
      const int r = q / KB;
      const int c = q % KB;
      const int g = base + r;
      const int gs = (g < I) ? g : (I - 1);
    load_A_lane:
      for (int l = 0; l < LANES; ++l) {
#pragma HLS UNROLL
        data_t a = A[gs][c * LANES + l];
        A_loc[r][c][l] = (g < I) ? a : (data_t)0;
        C_loc[r][c][l] = 0;
      }
    }

    // main MAC pipeline: 64 B blocks x (64 k-beats x 16 j) iterations,
    // next B block prefetched into the other ping-pong buffer
  mac_loop:
    for (int n = 0; n < NBLK * BLK_BEATS; ++n) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=B_buf0 inter false
#pragma HLS DEPENDENCE variable=B_buf1 inter false
#pragma HLS DEPENDENCE variable=C_loc inter RAW distance=16 true
      const int jb = n / BLK_BEATS;
      const int t  = n % BLK_BEATS;
      const int kc = t / JG;    // k beat used for compute
      const int jj = t % JG;    // j row within block
      const int r2 = t / KB;    // B row being prefetched (within block)
      const int c2 = t % KB;    // B k beat being prefetched
      const bool odd = (jb & 1);
      const int nb = (jb + 1) & (NBLK - 1);
      const int brow = nb * JG + r2;

      data_t bv[LANES];
#pragma HLS ARRAY_PARTITION variable=bv complete
    mac_bread:
      for (int l = 0; l < LANES; ++l) {
#pragma HLS UNROLL
        bv[l] = B[brow][c2 * LANES + l];
      }

      data_t bop[LANES];
#pragma HLS ARRAY_PARTITION variable=bop complete
    mac_bsel:
      for (int l = 0; l < LANES; ++l) {
#pragma HLS UNROLL
        data_t b0 = B_buf0[jj][kc][l];
        data_t b1 = B_buf1[jj][kc][l];
        bop[l] = odd ? b1 : b0;
      }
    mac_bfill:
      for (int l = 0; l < LANES; ++l) {
#pragma HLS UNROLL
        if (odd) B_buf0[r2][c2][l] = bv[l];
        else     B_buf1[r2][c2][l] = bv[l];
      }

    mac_rows:
      for (int i = 0; i < TI; ++i) {
#pragma HLS UNROLL
        data_t p[LANES];
#pragma HLS ARRAY_PARTITION variable=p complete
      mac_lanes:
        for (int l = 0; l < LANES; ++l) {
#pragma HLS UNROLL
          p[l] = A_loc[i][kc][l] * bop[l];
        }
        data_t s = tree16(p);
        C_loc[i][jb][jj] = C_loc[i][jb][jj] + s;
      }
    }

    // store: C[i][j] = C[i][j] + acc, row-wise burst read then burst write
  store_rows:
    for (int r = 0; r < TI; ++r) {
      const int g = base + r;
      if (g < I) {
      rd_C:
        for (int c = 0; c < JB; ++c) {
#pragma HLS PIPELINE II=1
        rd_C_lane:
          for (int l = 0; l < LANES; ++l) {
#pragma HLS UNROLL
            rowbuf[c][l] = C[g][c * LANES + l];
          }
        }
      wr_C:
        for (int c = 0; c < JB; ++c) {
#pragma HLS PIPELINE II=1
        wr_C_lane:
          for (int l = 0; l < LANES; ++l) {
#pragma HLS UNROLL
            C[g][c * LANES + l] = rowbuf[c][l] + C_loc[r][c][l];
          }
        }
      }
    }
  }
}