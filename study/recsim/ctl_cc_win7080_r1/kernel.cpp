#include "kernel.h"
#include <ap_int.h>
#include <hls_stream.h>

// ---------------------------------------------------------------------------
// Design parameters
// ---------------------------------------------------------------------------
#define LANES 16                    // floats per 512-bit beat
#define KB    (K / LANES)           // 64 beats per row
#define PI    86                    // parallel A rows (86*16 = 1376 fp32 MACs)
#define NB    ((I + PI - 1) / PI)   // 12 row blocks (last one partially padded)
#define JG    64                    // B rows per group (acc recurrence distance)
#define NG    (J / JG)              // 16 groups per row block
#define QB    (JG / LANES)          // 4 output beats per row per group
#define GW    (JG * KB)             // 4096 beats per B group

typedef ap_uint<512> wide_t;

struct col_t {
  float v[PI];
};

static inline float u2f(ap_uint<32> x) {
  union { unsigned u; float f; } c;
  c.u = (unsigned)x.to_uint();
  return c.f;
}

static inline ap_uint<32> f2u(float f) {
  union { unsigned u; float f; } c;
  c.f = f;
  return (ap_uint<32>)c.u;
}

// ---------------------------------------------------------------------------
// Load A row blocks (row-major beats), zero-padded rows beyond I
// ---------------------------------------------------------------------------
static void load_A(data_t A[I][K], hls::stream<wide_t>& a_s) {
load_A_block:
  for (int b = 0; b < NB; b++) {
    const int base = b * PI;
    const int nrow = (I - base < PI) ? (I - base) : PI;
  load_A_row:
    for (int r = 0; r < nrow; r++) {
#pragma HLS LOOP_TRIPCOUNT min=78 max=86 avg=85
    load_A_beat:
      for (int kb = 0; kb < KB; kb++) {
#pragma HLS PIPELINE II=1
        wide_t w;
      load_A_lane:
        for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
          w.range(32 * l + 31, 32 * l) = f2u(A[base + r][kb * LANES + l]);
        }
        a_s.write(w);
      }
    }
  pad_A_row:
    for (int r = nrow; r < PI; r++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=8 avg=1
    pad_A_beat:
      for (int kb = 0; kb < KB; kb++) {
#pragma HLS PIPELINE II=1
        a_s.write((wide_t)0);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Stream all of B (row-major beats) once per row block
// ---------------------------------------------------------------------------
static void load_B(data_t B[J][K], hls::stream<wide_t>& b_s) {
load_B_block:
  for (int b = 0; b < NB; b++) {
  load_B_row:
    for (int j = 0; j < J; j++) {
    load_B_beat:
      for (int kb = 0; kb < KB; kb++) {
#pragma HLS PIPELINE II=1
        wide_t w;
      load_B_lane:
        for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
          w.range(32 * l + 31, 32 * l) = f2u(B[j][kb * LANES + l]);
        }
        b_s.write(w);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Reorder each 64-row B group from (jj, kb) to (kb, jj) order, ping-pong
// ---------------------------------------------------------------------------
static void reorder_B(hls::stream<wide_t>& in_s, hls::stream<wide_t>& out_s) {
  wide_t bbuf[2 * GW];
#pragma HLS BIND_STORAGE variable=bbuf type=ram_s2p impl=uram
  const int NGT = NB * NG;
reorder_group:
  for (int g = 0; g <= NGT; g++) {
  reorder_beat:
    for (int w = 0; w < GW; w++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=bbuf type=inter dependent=false
#pragma HLS DEPENDENCE variable=bbuf type=intra dependent=false
      if (g < NGT) {
        bbuf[(g & 1) * GW + w] = in_s.read();        // w = jj*KB + kb
      }
      if (g > 0) {
        const int kb = w >> 6;                        // w = kb*JG + jj
        const int jj = w & (JG - 1);
        out_s.write(bbuf[((g - 1) & 1) * GW + jj * KB + kb]);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Compute: 86 rows x 16 k-lanes MAC per cycle, acc distance = 64 iterations
// ---------------------------------------------------------------------------
static void compute_mm(hls::stream<wide_t>& a_s, hls::stream<wide_t>& b_s,
                       hls::stream<col_t>& c_s) {
  wide_t abuf[2][PI][KB];
#pragma HLS ARRAY_PARTITION variable=abuf complete dim=2
  float acc[PI][JG];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1

  // prologue: first A block
prologue_row:
  for (int r = 0; r < PI; r++) {
  prologue_beat:
    for (int kb = 0; kb < KB; kb++) {
#pragma HLS PIPELINE II=1
      abuf[0][r][kb] = a_s.read();
    }
  }

compute_block:
  for (int b = 0; b < NB; b++) {
    const int cur = b & 1;
    const int nxt = cur ^ 1;
    const bool has_next = (b + 1) < NB;
  compute_main:
    for (int t = 0; t < NG * GW; t++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=acc type=inter dependent=true distance=64
#pragma HLS DEPENDENCE variable=abuf type=inter dependent=false
#pragma HLS DEPENDENCE variable=abuf type=intra dependent=false
      const int kb = (t >> 6) & (KB - 1);
      const int jj = t & (JG - 1);

      // overlapped load of next A block
      if (has_next && t < PI * KB) {
        abuf[nxt][t >> 6][t & (KB - 1)] = a_s.read();
      }

      const wide_t bw = b_s.read();
      float bv[LANES];
#pragma HLS ARRAY_PARTITION variable=bv complete
    unpack_b:
      for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
        bv[l] = u2f(bw.range(32 * l + 31, 32 * l));
      }

      const bool first = (kb == 0);
      const bool last = (kb == KB - 1);
      col_t outc;

    pe_row:
      for (int ii = 0; ii < PI; ii++) {
#pragma HLS UNROLL
        const wide_t aw = abuf[cur][ii][kb];
        float p[LANES];
#pragma HLS ARRAY_PARTITION variable=p complete
      pe_mul:
        for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
          p[l] = u2f(aw.range(32 * l + 31, 32 * l)) * bv[l];
        }
        float s8[8], s4[4], s2[2];
#pragma HLS ARRAY_PARTITION variable=s8 complete
#pragma HLS ARRAY_PARTITION variable=s4 complete
#pragma HLS ARRAY_PARTITION variable=s2 complete
      tree_l1:
        for (int m = 0; m < 8; m++) {
#pragma HLS UNROLL
          s8[m] = p[2 * m] + p[2 * m + 1];
        }
      tree_l2:
        for (int m = 0; m < 4; m++) {
#pragma HLS UNROLL
          s4[m] = s8[2 * m] + s8[2 * m + 1];
        }
      tree_l3:
        for (int m = 0; m < 2; m++) {
#pragma HLS UNROLL
          s2[m] = s4[2 * m] + s4[2 * m + 1];
        }
        const float s = s2[0] + s2[1];
        const float old = first ? 0.0f : acc[ii][jj];
        const float nv = old + s;
        acc[ii][jj] = nv;
        outc.v[ii] = nv;
      }
      if (last) {
        c_s.write(outc);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Store: C[i][j] = C[i][j] + sum, one read + one write per element, 512-bit
// ---------------------------------------------------------------------------
static void store_C(data_t C[I][J], hls::stream<col_t>& c_s) {
  float cb[JG][PI];
#pragma HLS ARRAY_PARTITION variable=cb cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=cb complete dim=2
  wide_t cbuf[PI * QB];

store_block:
  for (int b = 0; b < NB; b++) {
    const int base = b * PI;
    const int nrow = (I - base < PI) ? (I - base) : PI;
  store_group:
    for (int g = 0; g < NG; g++) {
    recv_col:
      for (int jj = 0; jj < JG; jj++) {
#pragma HLS PIPELINE II=1
        const col_t cv = c_s.read();
      recv_row:
        for (int ii = 0; ii < PI; ii++) {
#pragma HLS UNROLL
          cb[jj][ii] = cv.v[ii];
        }
      }
    read_c_row:
      for (int r = 0; r < nrow; r++) {
#pragma HLS LOOP_TRIPCOUNT min=78 max=86 avg=85
      read_c_beat:
        for (int q = 0; q < QB; q++) {
#pragma HLS PIPELINE II=1
          wide_t w;
        read_c_lane:
          for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
            const float c = C[base + r][g * JG + q * LANES + l];
            w.range(32 * l + 31, 32 * l) = f2u(c + cb[q * LANES + l][r]);
          }
          cbuf[r * QB + q] = w;
        }
      }
    write_c_row:
      for (int r = 0; r < nrow; r++) {
#pragma HLS LOOP_TRIPCOUNT min=78 max=86 avg=85
      write_c_beat:
        for (int q = 0; q < QB; q++) {
#pragma HLS PIPELINE II=1
          const wide_t w = cbuf[r * QB + q];
        write_c_lane:
          for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
            C[base + r][g * JG + q * LANES + l] = u2f(w.range(32 * l + 31, 32 * l));
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Top
// ---------------------------------------------------------------------------
extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16 depth=1048576
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16 depth=1048576
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16 max_write_burst_length=64 num_write_outstanding=16 depth=1048576
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS DATAFLOW
  hls::stream<wide_t> a_s("a_s");
  hls::stream<wide_t> b_raw("b_raw");
  hls::stream<wide_t> b_s("b_s");
  hls::stream<col_t> c_s("c_s");
#pragma HLS STREAM variable=a_s depth=64
#pragma HLS STREAM variable=b_raw depth=64
#pragma HLS STREAM variable=b_s depth=64
#pragma HLS STREAM variable=c_s depth=128

  load_A(A, a_s);
  load_B(B, b_raw);
  reorder_B(b_raw, b_s);
  compute_mm(a_s, b_s, c_s);
  store_C(C, c_s);
}