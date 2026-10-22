#include "kernel.h"
#include <hls_stream.h>

// ---------------------------------------------------------------------------
// Design constants
// ---------------------------------------------------------------------------
#define LANES 16                 // floats per 512-bit beat
#define TI    64                 // A rows resident on chip (PE rows)
#define JBW   16                 // output columns per block (= recurrence spacing)
#define KB    (K / LANES)        // 512-bit beats per row of length K   (64)
#define NT    (I / TI)           // number of row tiles                 (16)
#define NJB   (J / JBW)          // column blocks per row tile          (64)
#define BLK   (JBW * KB)         // compute iterations per block        (1024)
#define CB    (J / LANES)        // 512-bit beats per C row             (64)

struct beat_t {
  data_t v[LANES];
};

// 16 -> 1 balanced floating-point adder tree
static data_t tree16(const data_t p[LANES]) {
#pragma HLS INLINE
  data_t s8[8];
  data_t s4[4];
  data_t s2[2];
tree_lvl1:
  for (int i = 0; i < 8; i++) {
#pragma HLS UNROLL
    s8[i] = p[2 * i] + p[2 * i + 1];
  }
tree_lvl2:
  for (int i = 0; i < 4; i++) {
#pragma HLS UNROLL
    s4[i] = s8[2 * i] + s8[2 * i + 1];
  }
tree_lvl3:
  for (int i = 0; i < 2; i++) {
#pragma HLS UNROLL
    s2[i] = s4[2 * i] + s4[2 * i + 1];
  }
  return s2[0] + s2[1];
}

// ---------------------------------------------------------------------------
// Stage 1: stream all of B (row-major, 512-bit beats) once per row tile
// ---------------------------------------------------------------------------
static void load_B(data_t B[J][K], hls::stream<beat_t> &bs) {
ldb_tile:
  for (int t = 0; t < NT; t++) {
  ldb_beat:
    for (int x = 0; x < J * KB; x++) {
#pragma HLS PIPELINE II=1
      int j  = x / KB;
      int kb = x % KB;
      beat_t bt;
    ldb_lane:
      for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
        bt.v[l] = B[j][kb * LANES + l];
      }
      bs.write(bt);
    }
  }
}

// ---------------------------------------------------------------------------
// Stage 2: A/C tile staging + 1024-MAC/cycle compute + C store
// ---------------------------------------------------------------------------
static void compute(data_t A[I][K], data_t C[I][J], hls::stream<beat_t> &bs) {
  data_t A_loc[TI][K];
#pragma HLS ARRAY_PARTITION variable=A_loc complete dim=1
#pragma HLS ARRAY_PARTITION variable=A_loc cyclic factor=16 dim=2

  data_t Cl[TI][J];
#pragma HLS ARRAY_PARTITION variable=Cl complete dim=1
#pragma HLS ARRAY_PARTITION variable=Cl cyclic factor=16 dim=2

  data_t Bp[BLK][LANES];
#pragma HLS ARRAY_PARTITION variable=Bp complete dim=2
  data_t Bq[BLK][LANES];
#pragma HLS ARRAY_PARTITION variable=Bq complete dim=2

  data_t acc[TI][JBW];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1

  // Prologue: first B block (block 0 of tile 0) into the even buffer
prologue_b:
  for (int n = 0; n < BLK; n++) {
#pragma HLS PIPELINE II=1
    beat_t bt = bs.read();
  prologue_lane:
    for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
      Bp[n][l] = bt.v[l];
    }
  }

row_tile:
  for (int t = 0; t < NT; t++) {
    const int  i0        = t * TI;
    const bool last_tile = (t == NT - 1);

    // ---- load A tile and C tile (contiguous 512-bit bursts, two ports) ----
  load_ac:
    for (int x = 0; x < TI * CB; x++) {
#pragma HLS PIPELINE II=1
      int r = x / CB;
      int c = x % CB;
    load_ac_lane:
      for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
        A_loc[r][c * LANES + l] = A[i0 + r][c * LANES + l];
        Cl[r][c * LANES + l]    = C[i0 + r][c * LANES + l];
      }
    }

    // ---- main compute: flat loop over all column blocks of this tile ----
  main_mac:
    for (int g = 0; g < NJB * BLK; g++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=acc inter RAW distance=16 true
#pragma HLS DEPENDENCE variable=Cl inter false
#pragma HLS DEPENDENCE variable=Bp inter false
#pragma HLS DEPENDENCE variable=Bq inter false
      const int  jb      = g / BLK;
      const int  n       = g % BLK;
      const int  kb      = n / JBW;
      const int  jj      = n % JBW;
      const int  ra      = jj * KB + kb;
      const bool cur_odd = (jb & 1);
      const bool fill    = !(last_tile && (jb == NJB - 1));
      const bool first16 = (n < JBW);
      const bool drain   = first16 && (jb > 0);

      // prefetch next block of B into the idle ping-pong buffer
      beat_t nb;
      if (fill) {
        nb = bs.read();
      }

      data_t b[LANES];
#pragma HLS ARRAY_PARTITION variable=b complete
    rd_b_lane:
      for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
        b[l] = cur_odd ? Bq[ra][l] : Bp[ra][l];
      }

      if (fill) {
      wr_b_lane:
        for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
          if (cur_odd) {
            Bp[n][l] = nb.v[l];
          } else {
            Bq[n][l] = nb.v[l];
          }
        }
      }

    pe_row:
      for (int r = 0; r < TI; r++) {
#pragma HLS UNROLL
        data_t p[LANES];
#pragma HLS ARRAY_PARTITION variable=p complete
      pe_lane:
        for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
          p[l] = A_loc[r][kb * LANES + l] * b[l];
        }
        data_t s    = tree16(p);
        data_t head = acc[r][jj];
        data_t sum  = head + s;
        acc[r][jj]  = first16 ? s : sum;
        // first 16 iterations of block jb retire block jb-1, column jj (= n)
        if (drain) {
          int col = (jb - 1) * JBW + n;
          Cl[r][col] = Cl[r][col] + head;
        }
      }
    }

    // ---- retire last column block of the tile ----
  tail_drain:
    for (int jj = 0; jj < JBW; jj++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=Cl inter false
    tail_row:
      for (int r = 0; r < TI; r++) {
#pragma HLS UNROLL
        int col = (NJB - 1) * JBW + jj;
        Cl[r][col] = Cl[r][col] + acc[r][jj];
      }
    }

    // ---- store C tile (contiguous 512-bit bursts) ----
  store_c:
    for (int x = 0; x < TI * CB; x++) {
#pragma HLS PIPELINE II=1
      int r = x / CB;
      int c = x % CB;
    store_c_lane:
      for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
        C[i0 + r][c * LANES + l] = Cl[r][c * LANES + l];
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Top
// ---------------------------------------------------------------------------
extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=16 num_write_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=16 num_write_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=16 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS DATAFLOW
  hls::stream<beat_t> bs("b_stream");
#pragma HLS STREAM variable=bs depth=64

  load_B(B, bs);
  compute(A, C, bs);
}