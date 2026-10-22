#include "kernel.h"
#include <hls_stream.h>

// ---------------------------------------------------------------------------
// Datapath sizing: PI x PJ = 21 x 64 = 1344 fp32 MACs per cycle
//   fmul (max_dsp, 3 DSP) + fadd (full_dsp, 2 DSP) = 5 DSP/MAC -> ~6720 DSP
// Outer-product formulation per tile of PI rows of C:
//   for k: for jb (J/PJ = 16 column blocks): acc[r][jb*PJ+p] += A[r][k]*B[jb*PJ+p][k]
// Each accumulator is revisited every 16 pipeline iterations (> fadd latency).
// ---------------------------------------------------------------------------
#define LANES       16                      // 512-bit beat / 32-bit float
#define PI          21                      // rows of C per tile (parallel)
#define PJ          64                      // columns of C per cycle (parallel)
#define NJB         (J / PJ)                // 16 column blocks
#define NT          ((I + PI - 1) / PI)     // 49 row tiles (last one guarded)
#define KB          (K / LANES)             // 64 beats per A/B row
#define JBEATS      (J / LANES)             // 64 beats per C row
#define TILE_BEATS  (PI * KB)               // 1344 A beats per tile
#define TILE_OBEATS (PI * JBEATS)           // 1344 C beats per tile
#define CITERS      (K * NJB)               // 16384 compute iterations per tile

struct pkt_t {
  data_t d[LANES];
};

// ---------------------------------------------------------------------------
// Stage 1: stream A row tiles (512-bit bursts, one row = 64 beats)
// ---------------------------------------------------------------------------
static void load_A(data_t A[I][K], hls::stream<pkt_t> &a_s) {
load_A_tile:
  for (int t = 0; t < NT; t++) {
  load_A_row:
    for (int r = 0; r < PI; r++) {
      int row = t * PI + r;
      if (row < I) {
      load_A_beat:
        for (int kb = 0; kb < KB; kb++) {
#pragma HLS PIPELINE II=1
          pkt_t p;
        load_A_lane:
          for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
            p.d[l] = A[row][kb * LANES + l];
          }
          a_s.write(p);
        }
      } else {
      pad_A_beat:
        for (int kb = 0; kb < KB; kb++) {
#pragma HLS PIPELINE II=1
          pkt_t p;
        pad_A_lane:
          for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
            p.d[l] = 0.0f;
          }
          a_s.write(p);
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Stage 2: B resident on chip, 21x64 MAC array, overlapped A receive / C drain
// ---------------------------------------------------------------------------
static void compute(data_t B[J][K], hls::stream<pkt_t> &a_s, hls::stream<pkt_t> &o_s) {
  // Bbuf[p][jb][k] = B[jb*PJ + p][k]
  static data_t Bbuf[PJ][NJB][K];
#pragma HLS ARRAY_PARTITION variable=Bbuf complete dim=1
#pragma HLS ARRAY_PARTITION variable=Bbuf cyclic factor=16 dim=3
#pragma HLS BIND_STORAGE variable=Bbuf type=ram_s2p impl=bram

  // ping-pong A tile buffers
  static data_t Abuf[2][PI][K];
#pragma HLS ARRAY_PARTITION variable=Abuf complete dim=1
#pragma HLS ARRAY_PARTITION variable=Abuf complete dim=2
#pragma HLS ARRAY_PARTITION variable=Abuf cyclic factor=16 dim=3
#pragma HLS BIND_STORAGE variable=Abuf type=ram_s2p impl=lutram

  // accumulators and finished-tile output buffer: banks (r, j%64), depth 16
  static data_t acc[PI][J];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc cyclic factor=64 dim=2
#pragma HLS BIND_STORAGE variable=acc type=ram_s2p impl=lutram

  static data_t obuf[PI][J];
#pragma HLS ARRAY_PARTITION variable=obuf complete dim=1
#pragma HLS ARRAY_PARTITION variable=obuf cyclic factor=64 dim=2
#pragma HLS BIND_STORAGE variable=obuf type=ram_s2p impl=lutram

  // ---- load B once (512-bit bursts) ----
load_B_row:
  for (int j = 0; j < J; j++) {
  load_B_beat:
    for (int kb = 0; kb < KB; kb++) {
#pragma HLS PIPELINE II=1
    load_B_lane:
      for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
        Bbuf[j % PJ][j / PJ][kb * LANES + l] = B[j][kb * LANES + l];
      }
    }
  }

  // ---- prologue: receive A tile 0 ----
recv_A0_beat:
  for (int b = 0; b < TILE_BEATS; b++) {
#pragma HLS PIPELINE II=1
    pkt_t p = a_s.read();
    int r = b / KB;
    int kb = b % KB;
  recv_A0_lane:
    for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
      Abuf[0][r][kb * LANES + l] = p.d[l];
    }
  }

  // ---- steady state: compute tile t, receive A(t+1), drain C(t-1) ----
compute_tile:
  for (int t = 0; t < NT; t++) {
    const int cb = t & 1;
    const int nb = cb ^ 1;
    const bool has_next = (t + 1) < NT;
    const bool has_prev = t > 0;

  compute_it:
    for (int it = 0; it < CITERS; it++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=acc array inter RAW distance=16 true
#pragma HLS DEPENDENCE variable=obuf inter false
#pragma HLS DEPENDENCE variable=obuf intra false
#pragma HLS DEPENDENCE variable=Abuf inter false
#pragma HLS DEPENDENCE variable=Abuf intra false
      const int k = it / NJB;
      const int jb = it % NJB;

      // receive next A tile into the idle ping-pong buffer
      if (has_next && it < TILE_BEATS) {
        pkt_t p = a_s.read();
        int r = it / KB;
        int kb = it % KB;
      recv_A_lane:
        for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
          Abuf[nb][r][kb * LANES + l] = p.d[l];
        }
      }

      // drain previous tile's results
      if (has_prev && it < TILE_OBEATS) {
        pkt_t o;
        int r = it / JBEATS;
        int c = it % JBEATS;
      drain_lane:
        for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
          o.d[l] = obuf[r][c * LANES + l];
        }
        o_s.write(o);
      }

      // operand fetch
      data_t a_reg[PI];
#pragma HLS ARRAY_PARTITION variable=a_reg complete dim=1
      data_t b_reg[PJ];
#pragma HLS ARRAY_PARTITION variable=b_reg complete dim=1
    fetch_a:
      for (int r = 0; r < PI; r++) {
#pragma HLS UNROLL
        a_reg[r] = Abuf[cb][r][k];
      }
    fetch_b:
      for (int p = 0; p < PJ; p++) {
#pragma HLS UNROLL
        b_reg[p] = Bbuf[p][jb][k];
      }

      // 21 x 64 MAC array
    mac_r:
      for (int r = 0; r < PI; r++) {
#pragma HLS UNROLL
      mac_p:
        for (int p = 0; p < PJ; p++) {
#pragma HLS UNROLL
          data_t prod = a_reg[r] * b_reg[p];
#pragma HLS BIND_OP variable=prod op=fmul impl=maxdsp
          data_t old = acc[r][jb * PJ + p];
          data_t base = (k == 0) ? (data_t)0.0f : old;
          data_t s = base + prod;
#pragma HLS BIND_OP variable=s op=fadd impl=fulldsp
          acc[r][jb * PJ + p] = s;
          if (k == K - 1) obuf[r][jb * PJ + p] = s;
        }
      }
    }
  }

  // ---- epilogue: drain last tile ----
drain_last_beat:
  for (int b = 0; b < TILE_OBEATS; b++) {
#pragma HLS PIPELINE II=1
    pkt_t o;
    int r = b / JBEATS;
    int c = b % JBEATS;
  drain_last_lane:
    for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
      o.d[l] = obuf[r][c * LANES + l];
    }
    o_s.write(o);
  }
}

// ---------------------------------------------------------------------------
// Stage 3: C[row] = C_old[row] + sum, 512-bit read and write bursts
// ---------------------------------------------------------------------------
static void store_C(data_t C[I][J], hls::stream<pkt_t> &o_s) {
  data_t cbuf[J];
#pragma HLS ARRAY_PARTITION variable=cbuf cyclic factor=16 dim=1

store_tile:
  for (int t = 0; t < NT; t++) {
  store_row:
    for (int r = 0; r < PI; r++) {
      int row = t * PI + r;
      if (row < I) {
      read_C_beat:
        for (int c = 0; c < JBEATS; c++) {
#pragma HLS PIPELINE II=1
        read_C_lane:
          for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
            cbuf[c * LANES + l] = C[row][c * LANES + l];
          }
        }
      write_C_beat:
        for (int c = 0; c < JBEATS; c++) {
#pragma HLS PIPELINE II=1
          pkt_t o = o_s.read();
        write_C_lane:
          for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
            C[row][c * LANES + l] = cbuf[c * LANES + l] + o.d[l];
          }
        }
      } else {
      discard_beat:
        for (int c = 0; c < JBEATS; c++) {
#pragma HLS PIPELINE II=1
          pkt_t o = o_s.read();
          (void)o;
        }
      }
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

#pragma HLS DATAFLOW
  hls::stream<pkt_t> a_s;
#pragma HLS STREAM variable=a_s depth=2048
  hls::stream<pkt_t> o_s;
#pragma HLS STREAM variable=o_s depth=2048

  load_A(A, a_s);
  compute(B, a_s, o_s);
  store_C(C, o_s);
}