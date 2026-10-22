#include "kernel.h"
#include <hls_stream.h>
#include <ap_int.h>

// C = A * B^T, output-stationary PE array of PI x PJ fp32 MACs.
// Each output accumulates over k in exactly the reference order (0 + p0 + p1 + ...),
// so results are bit-identical to the plain C loop.
//   - B (all J rows) is held on chip in bbuf, banked by j % PJ, one 512-bit word = 16 k values.
//   - A rows are processed in blocks of PI rows; per k the PE array sweeps the S = J / PJ
//     column groups (II=1), so an accumulator is revisited every S cycles (> fadd latency).
//   - B arrives k-chunk-major while the first group runs. That group holds QG row blocks
//     (sweep of QG*S cycles per k), which consumes B exactly at the AXI rate, so the B load
//     is hidden behind compute. Later groups are single blocks.
//   - Final sums go to cbuf at k = K-1 and are drained to store_C during the next group.
//   - compute is one flattened pipelined loop, so every on-chip memory has a single user.

#define PI 28                         // PE rows (A rows per block)
#define PJ 64                         // PE cols (B rows per column group)
#define LANES 16                      // floats per 512-bit AXI beat
#define S (J / PJ)                    // column groups per k
#define NB ((I + PI - 1) / PI)        // row blocks (last one padded)
#define QG 4                          // row blocks in the first group
#define NG (NB - QG + 1)              // groups
#define KW (K / LANES)                // AXI words per row
#define CH (PJ / LANES)               // AXI words per PE-row segment
#define DR (PI * S * CH)              // drain words per block
#define PRE 2                         // B chunks loaded before compute starts
#define DLY 64                        // drain start delay (> pipeline depth: cbuf RAW)
#define TOTAL (PRE * J + NB * K * S + DLY + DR + 8)

typedef ap_uint<32 * LANES> wide_t;

struct acol_t { data_t v[PI]; };
struct cvec_t { data_t v[LANES]; };

static inline data_t bits2f(ap_uint<32> x) {
  union { unsigned int u; float f; } c;
  c.u = (unsigned int)x;
  return c.f;
}
static inline ap_uint<32> f2bits(data_t f) {
  union { unsigned int u; float f; } c;
  c.f = f;
  return c.u;
}

// B k-chunk-major: all rows of chunk 0, then chunk 1, ...
static void load_B(data_t B[J][K], hls::stream<wide_t> &bs) {
  for (int kw = 0; kw < KW; kw++)
    for (int j = 0; j < J; j++) {
#pragma HLS PIPELINE II=1
      wide_t w;
      for (int u = 0; u < LANES; u++) {
#pragma HLS UNROLL
        w.range(32 * u + 31, 32 * u) = f2bits(B[j][kw * LANES + u]);
      }
      bs.write(w);
    }
}

// Per group and k-chunk: fetch the chunk of every row of the group, then emit
// the A columns in compute order (k, block q).
static void load_A(data_t A[I][K], hls::stream<acol_t> &as) {
  data_t abuf[QG][PI][LANES];
#pragma HLS ARRAY_PARTITION variable=abuf complete dim=2
#pragma HLS ARRAY_PARTITION variable=abuf complete dim=3
  for (int g = 0; g < NG; g++) {
    int nq = (g == 0) ? QG : 1;
    int base = (g == 0) ? 0 : (QG + g - 1) * PI;
    for (int kw = 0; kw < KW; kw++) {
      for (int r = 0; r < nq * PI; r++) {
#pragma HLS LOOP_TRIPCOUNT min=PI max=QG*PI
#pragma HLS PIPELINE II=1
        // padding rows past I re-read row I-1; their results are dropped by store_C
        int row = (base + r < I) ? base + r : I - 1;
        for (int u = 0; u < LANES; u++) {
#pragma HLS UNROLL
          abuf[r / PI][r % PI][u] = A[row][kw * LANES + u];
        }
      }
      for (int t = 0; t < LANES * nq; t++) {
#pragma HLS LOOP_TRIPCOUNT min=LANES max=QG*LANES
#pragma HLS PIPELINE II=1
        int kk = t / nq;
        int q = t % nq;
        acol_t c;
        for (int pi = 0; pi < PI; pi++) {
#pragma HLS UNROLL
          c.v[pi] = abuf[q][pi][kk];
        }
        as.write(c);
      }
    }
  }
}

static void compute(hls::stream<wide_t> &bs, hls::stream<acol_t> &as, hls::stream<cvec_t> &cs) {
  wide_t bbuf[PJ][S * KW];
#pragma HLS ARRAY_PARTITION variable=bbuf complete dim=1
#pragma HLS BIND_STORAGE variable=bbuf type=ram_2p impl=bram
  data_t acc[PI][PJ][QG * S];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc complete dim=2
#pragma HLS BIND_STORAGE variable=acc type=ram_2p impl=lutram
  data_t cbuf[PI][PJ][QG * S];
#pragma HLS ARRAY_PARTITION variable=cbuf complete dim=1
#pragma HLS ARRAY_PARTITION variable=cbuf complete dim=2
#pragma HLS BIND_STORAGE variable=cbuf type=ram_2p impl=lutram

  // B write state
  int bj = 0, bkw = 0;
  bool bdone = false;
  // compute state: k, position sq in the sweep of sqn = nq * S cycles
  int k = 0, sq = 0, sqn = QG * S, g = 0;
  bool cdone = false;
  // drain state: wait, words left, word index d = ((q * PI + pi) * S + s2) * CH + h
  int dwait = 0, dleft = 0, dq = 0, dpi = 0, dlo = 0;
  acol_t acol;

  for (int it = 0; it < TOTAL; it++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=bbuf inter false
#pragma HLS DEPENDENCE variable=acc inter false
#pragma HLS DEPENDENCE variable=cbuf inter false
    if (!bdone) {
      bbuf[bj % PJ][(bj / PJ) * KW + bkw] = bs.read();
      if (bj == J - 1) {
        bj = 0;
        if (bkw == KW - 1) bdone = true;
        bkw++;
      } else {
        bj++;
      }
    }

    if (it >= PRE * J && !cdone) {
      int s = sq % S;
      if (s == 0) acol = as.read();
      data_t bv[PJ];
#pragma HLS ARRAY_PARTITION variable=bv complete
      for (int pj = 0; pj < PJ; pj++) {
#pragma HLS UNROLL
        wide_t w = bbuf[pj][s * KW + k / LANES];
        int l = k % LANES;
        bv[pj] = bits2f(w.range(32 * l + 31, 32 * l));
      }
      for (int pi = 0; pi < PI; pi++) {
#pragma HLS UNROLL
        for (int pj = 0; pj < PJ; pj++) {
#pragma HLS UNROLL
          data_t p = acol.v[pi] * bv[pj];
          data_t old = (k == 0) ? (data_t)0 : acc[pi][pj][sq];
          data_t nv = old + p;
#pragma HLS BIND_OP variable=nv op=fadd impl=fulldsp latency=4
          acc[pi][pj][sq] = nv;
          if (k == K - 1) cbuf[pi][pj][sq] = nv;
        }
      }
      if (sq == sqn - 1) {
        sq = 0;
        if (k == K - 1) {
          k = 0;
          dwait = DLY;
          dleft = (sqn / S) * DR;
          dq = 0;
          dpi = 0;
          dlo = 0;
          sqn = S;
          if (g == NG - 1) cdone = true;
          g++;
        } else {
          k++;
        }
      } else {
        sq++;
      }
    }

    if (dwait > 0) {
      dwait--;
    } else if (dleft > 0) {
      int s2 = dlo / CH;
      int h = dlo % CH;
      cvec_t o;
      for (int u = 0; u < LANES; u++) {
#pragma HLS UNROLL
        o.v[u] = cbuf[dpi][h * LANES + u][dq * S + s2];
      }
      cs.write(o);
      dleft--;
      if (dlo == S * CH - 1) {
        dlo = 0;
        if (dpi == PI - 1) {
          dpi = 0;
          dq++;
        } else {
          dpi++;
        }
      } else {
        dlo++;
      }
    }
  }
}

static void store_C(hls::stream<cvec_t> &cs, data_t C[I][J]) {
  // rows arrive in order 0, 1, ..., I-1, then the padding rows of the last block
  for (int i = 0; i < I; i++)
    for (int v = 0; v < J / LANES; v++) {
#pragma HLS PIPELINE II=1
      cvec_t o = cs.read();
      for (int u = 0; u < LANES; u++) {
#pragma HLS UNROLL
        C[i][v * LANES + u] = o.v[u];
      }
    }
  for (int t = 0; t < (NB * PI - I) * (J / LANES); t++) {
#pragma HLS PIPELINE II=1
    cs.read();
  }
}

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A bundle=gmem0 depth=1048576 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=B bundle=gmem1 depth=1048576 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=C bundle=gmem2 depth=1048576 max_widen_bitwidth=512
#pragma HLS DATAFLOW
  hls::stream<wide_t> bs("bs");
  hls::stream<acol_t> as("as");
  hls::stream<cvec_t> cs("cs");
#pragma HLS STREAM variable=bs depth=4
#pragma HLS STREAM variable=as depth=8
#pragma HLS STREAM variable=cs depth=64
  load_B(B, bs);
  load_A(A, as);
  compute(bs, as, cs);
  store_C(cs, C);
}
