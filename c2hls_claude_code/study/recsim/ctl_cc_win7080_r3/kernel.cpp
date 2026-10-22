#include "kernel.h"
#include <hls_stream.h>

// ---------------------------------------------------------------------------
// Parallel datapath sizing (fp32 MAC ~= 3 DSP fmul + 2 DSP fadd)
//   TPI (rows of C) x TPJ (cols of C) x KL (k-lanes) = 4 x 21 x 16 = 1344 MACs
//   -> ~6720 DSP (inside 6317..7219 window)
// ---------------------------------------------------------------------------
#define TPI   4
#define TPJ   21
#define KL    16                         // floats per 512-bit beat
#define KB    (K / KL)                   // 64 k-beats per row
#define JSUB  ((J + TPJ - 1) / TPJ)      // 49 column groups
#define JP    (JSUB * TPJ)               // 1029 padded B rows
#define NT    (I / TPI)                  // 256 row tiles
#define CB    (J / KL)                   // 64 beats per C row

struct vec16_t { data_t d[KL]; };
struct avec_t  { data_t d[TPI][KL]; };
struct ovec_t  { data_t d[TPI][TPJ]; };

static data_t tree16(const data_t p[KL]) {
#pragma HLS INLINE
  data_t s8[8], s4[4], s2[2];
#pragma HLS ARRAY_PARTITION variable=s8 complete dim=1
#pragma HLS ARRAY_PARTITION variable=s4 complete dim=1
#pragma HLS ARRAY_PARTITION variable=s2 complete dim=1
  tree_l8: for (int t = 0; t < 8; ++t) {
#pragma HLS UNROLL
    s8[t] = p[2 * t] + p[2 * t + 1];
  }
  tree_l4: for (int t = 0; t < 4; ++t) {
#pragma HLS UNROLL
    s4[t] = s8[2 * t] + s8[2 * t + 1];
  }
  tree_l2: for (int t = 0; t < 2; ++t) {
#pragma HLS UNROLL
    s2[t] = s4[2 * t] + s4[2 * t + 1];
  }
  return s2[0] + s2[1];
}

// ---------------------------------------------------------------------------
// Load B once, row-major, one 512-bit beat per cycle
// ---------------------------------------------------------------------------
static void load_B_task(data_t B[J][K], hls::stream<vec16_t> &sB) {
  load_B_beats: for (int it = 0; it < J * KB; ++it) {
#pragma HLS PIPELINE II=1
    const int row = it >> 6;
    const int kb  = it & (KB - 1);
    vec16_t v;
    load_B_lane: for (int l = 0; l < KL; ++l) {
#pragma HLS UNROLL
      v.d[l] = B[row][kb * KL + l];
    }
    sB.write(v);
  }
}

// ---------------------------------------------------------------------------
// Load A row tiles (TPI contiguous rows) in bursts, emit k-beat slices
// ---------------------------------------------------------------------------
static void load_A_task(data_t A[I][K], hls::stream<avec_t> &sA) {
  data_t abuf[TPI][KB][KL];
#pragma HLS ARRAY_PARTITION variable=abuf complete dim=1
#pragma HLS ARRAY_PARTITION variable=abuf complete dim=3

  load_A_tiles: for (int t = 0; t < NT; ++t) {
    const int i0 = t * TPI;
    read_A_beats: for (int it = 0; it < TPI * KB; ++it) {
#pragma HLS PIPELINE II=1
      const int r  = it >> 6;
      const int kb = it & (KB - 1);
      read_A_lane: for (int l = 0; l < KL; ++l) {
#pragma HLS UNROLL
        abuf[r][kb][l] = A[i0 + r][kb * KL + l];
      }
    }
    emit_A_beats: for (int kb = 0; kb < KB; ++kb) {
#pragma HLS PIPELINE II=1
      avec_t av;
      emit_A_row: for (int r = 0; r < TPI; ++r) {
#pragma HLS UNROLL
        emit_A_lane: for (int l = 0; l < KL; ++l) {
#pragma HLS UNROLL
          av.d[r][l] = abuf[r][kb][l];
        }
      }
      sA.write(av);
    }
  }
}

// ---------------------------------------------------------------------------
// Compute: B resident on chip (URAM), 4x21 PE array, 16-lane k tree per PE,
// interleaved accumulators (reuse distance JSUB=49 iterations).
// ---------------------------------------------------------------------------
static void compute_task(hls::stream<vec16_t> &sB, hls::stream<avec_t> &sA,
                         hls::stream<ovec_t> &sO) {
  data_t B_loc[TPJ][KL][JSUB * KB];
#pragma HLS ARRAY_PARTITION variable=B_loc complete dim=1
#pragma HLS ARRAY_PARTITION variable=B_loc complete dim=2
#pragma HLS BIND_STORAGE variable=B_loc type=ram_2p impl=uram

  data_t acc[TPI][TPJ][JSUB];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc complete dim=2

  data_t areg[TPI][KL];
#pragma HLS ARRAY_PARTITION variable=areg complete dim=0

  // ---- fill resident B (padded rows are zero) ----
  int row = 0, kbf = 0, pjf = 0, jsf = 0;
  fill_B_loc: for (int it = 0; it < JP * KB; ++it) {
#pragma HLS PIPELINE II=1
    vec16_t v;
    if (row < J) {
      v = sB.read();
    } else {
      zero_lane: for (int l = 0; l < KL; ++l) {
#pragma HLS UNROLL
        v.d[l] = (data_t)0;
      }
    }
    fill_B_lane: for (int l = 0; l < KL; ++l) {
#pragma HLS UNROLL
      B_loc[pjf][l][jsf * KB + kbf] = v.d[l];
    }
    if (kbf == KB - 1) {
      kbf = 0;
      row++;
      if (pjf == TPJ - 1) { pjf = 0; jsf++; } else { pjf++; }
    } else {
      kbf++;
    }
  }

  // ---- main MAC pipeline: (tile, k-beat, column-group) flattened ----
  int js = 0, kc = 0;
  mac_main: for (int it = 0; it < NT * KB * JSUB; ++it) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=acc inter false
    if (js == 0) {
      avec_t av = sA.read();
      a_copy_r: for (int r = 0; r < TPI; ++r) {
#pragma HLS UNROLL
        a_copy_l: for (int l = 0; l < KL; ++l) {
#pragma HLS UNROLL
          areg[r][l] = av.d[r][l];
        }
      }
    }
    const int baddr = js * KB + kc;
    ovec_t ov;
    pe_i: for (int pi = 0; pi < TPI; ++pi) {
#pragma HLS UNROLL
      pe_j: for (int pj = 0; pj < TPJ; ++pj) {
#pragma HLS UNROLL
        data_t prod[KL];
#pragma HLS ARRAY_PARTITION variable=prod complete dim=1
        mul_lane: for (int l = 0; l < KL; ++l) {
#pragma HLS UNROLL
          prod[l] = areg[pi][l] * B_loc[pj][l][baddr];
        }
        data_t s    = tree16(prod);
        data_t prev = (kc == 0) ? (data_t)0 : acc[pi][pj][js];
        data_t nv   = prev + s;
        acc[pi][pj][js] = nv;
        ov.d[pi][pj]    = nv;
      }
    }
    if (kc == KB - 1) sO.write(ov);

    if (js == JSUB - 1) {
      js = 0;
      kc = (kc == KB - 1) ? 0 : kc + 1;
    } else {
      js++;
    }
  }
}

// ---------------------------------------------------------------------------
// Store: read C tile (bursts), receive results, C = C + dot, write (bursts)
// ---------------------------------------------------------------------------
static void store_C_task(hls::stream<ovec_t> &sO, data_t C[I][J]) {
  data_t cbuf[TPI][J];
#pragma HLS ARRAY_PARTITION variable=cbuf complete dim=1
#pragma HLS ARRAY_PARTITION variable=cbuf cyclic factor=16 dim=2
  data_t obuf[TPI][TPJ][JSUB];
#pragma HLS ARRAY_PARTITION variable=obuf complete dim=1
#pragma HLS ARRAY_PARTITION variable=obuf complete dim=2

  store_tiles: for (int t = 0; t < NT; ++t) {
    const int i0 = t * TPI;

    read_C_beats: for (int it = 0; it < TPI * CB; ++it) {
#pragma HLS PIPELINE II=1
      const int r  = it >> 6;
      const int cb = it & (CB - 1);
      read_C_lane: for (int l = 0; l < KL; ++l) {
#pragma HLS UNROLL
        cbuf[r][cb * KL + l] = C[i0 + r][cb * KL + l];
      }
    }

    recv_out: for (int js = 0; js < JSUB; ++js) {
#pragma HLS PIPELINE II=1
      ovec_t ov = sO.read();
      recv_r: for (int pi = 0; pi < TPI; ++pi) {
#pragma HLS UNROLL
        recv_c: for (int pj = 0; pj < TPJ; ++pj) {
#pragma HLS UNROLL
          obuf[pi][pj][js] = ov.d[pi][pj];
        }
      }
    }

    int b0 = 0, a0 = 0;
    write_C_beats: for (int it = 0; it < TPI * CB; ++it) {
#pragma HLS PIPELINE II=1
      const int r  = it >> 6;
      const int cb = it & (CB - 1);
      if (cb == 0) { b0 = 0; a0 = 0; }
      data_t val[TPJ];
#pragma HLS ARRAY_PARTITION variable=val complete dim=1
      gather_bank: for (int p = 0; p < TPJ; ++p) {
#pragma HLS UNROLL
        int addr = (p >= b0) ? a0 : (a0 + 1);
        if (addr > JSUB - 1) addr = JSUB - 1;
        val[p] = obuf[r][p][addr];
      }
      write_C_lane: for (int l = 0; l < KL; ++l) {
#pragma HLS UNROLL
        int sel = b0 + l;
        if (sel >= TPJ) sel -= TPJ;
        C[i0 + r][cb * KL + l] = cbuf[r][cb * KL + l] + val[sel];
      }
      b0 += KL;
      if (b0 >= TPJ) { b0 -= TPJ; a0++; }
    }
  }
}

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  hls::stream<vec16_t> sB("sB");
#pragma HLS STREAM variable=sB depth=64
  hls::stream<avec_t> sA("sA");
#pragma HLS STREAM variable=sA depth=128
  hls::stream<ovec_t> sO("sO");
#pragma HLS STREAM variable=sO depth=64

#pragma HLS DATAFLOW
  load_B_task(B, sB);
  load_A_task(A, sA);
  compute_task(sB, sA, sO);
  store_C_task(sO, C);
}