#include "kernel.h"
#include <hls_stream.h>
#include <ap_int.h>

// C = A * B^T, output-stationary PE array of PI x PJ fp32 MACs.
// Each output accumulates over k in exactly the reference order (0 + p0 + p1 + ...),
// so results are bit-identical to the plain C loop.
//   - B (all 1024 rows) is held on chip, banked by j % PJ.
//   - A is streamed per block of PI rows, one column (PI values) per k.
//   - Per k, the S = J / PJ column groups are swept in a pipelined loop: the
//     accumulator dependency distance is S cycles, which hides the fadd latency.
//   - Results of block b are drained to the store process while block b+1 computes.

#define LANES 16                      // floats per 512-bit AXI beat
#define PI 32                         // PE rows (A rows per block)
#define PJ 32                         // PE cols (B rows per group)
#define S (J / PJ)                    // column groups per k
#define NB ((I + PI - 1) / PI)        // row blocks
#define KV (K / LANES)                // 512-bit words per row

typedef ap_uint<512> wide_t;

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

static void load_B(data_t B[J][K], hls::stream<wide_t> &bs) {
  for (int j = 0; j < J; j++)
    for (int kv = 0; kv < KV; kv++) {
#pragma HLS PIPELINE II=1
      wide_t w;
      for (int u = 0; u < LANES; u++) {
#pragma HLS UNROLL
        w.range(32 * u + 31, 32 * u) = f2bits(B[j][kv * LANES + u]);
      }
      bs.write(w);
    }
}

static void load_A(data_t A[I][K], hls::stream<acol_t> &as) {
  data_t abuf[PI][K];
#pragma HLS ARRAY_PARTITION variable=abuf complete dim=1
#pragma HLS ARRAY_PARTITION variable=abuf cyclic factor=LANES dim=2
  for (int b = 0; b < NB; b++) {
    for (int r = 0; r < PI; r++)
      for (int kv = 0; kv < KV; kv++) {
#pragma HLS PIPELINE II=1
        int row = b * PI + r;
        for (int u = 0; u < LANES; u++) {
#pragma HLS UNROLL
          abuf[r][kv * LANES + u] = (row < I) ? A[row][kv * LANES + u] : (data_t)0;
        }
      }
    for (int k = 0; k < K; k++) {
#pragma HLS PIPELINE II=1
      acol_t c;
      for (int r = 0; r < PI; r++) {
#pragma HLS UNROLL
        c.v[r] = abuf[r][k];
      }
      as.write(c);
    }
  }
}

static void compute(hls::stream<wide_t> &bs, hls::stream<acol_t> &as, hls::stream<cvec_t> &cs) {
  wide_t bbuf[PJ][S * KV];
#pragma HLS ARRAY_PARTITION variable=bbuf complete dim=1
#pragma HLS BIND_STORAGE variable=bbuf type=ram_2p impl=uram
  data_t acc[2][PI][PJ][S];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc complete dim=2
#pragma HLS ARRAY_PARTITION variable=acc complete dim=3
#pragma HLS BIND_STORAGE variable=acc type=ram_t2p impl=lutram

  for (int j = 0; j < J; j++)
    for (int kv = 0; kv < KV; kv++) {
#pragma HLS PIPELINE II=1
      bbuf[j % PJ][(j / PJ) * KV + kv] = bs.read();
    }

  acol_t acol;
  for (int b = 0; b < NB; b++) {
    int cur = b & 1;
    int prv = cur ^ 1;
    for (int t = 0; t < K * S; t++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=acc inter false
      int k = t / S;
      int s = t % S;
      if (s == 0) acol = as.read();
      data_t bv[PJ];
#pragma HLS ARRAY_PARTITION variable=bv complete
      for (int pj = 0; pj < PJ; pj++) {
#pragma HLS UNROLL
        wide_t w = bbuf[pj][s * KV + k / LANES];
        int l = k % LANES;
        bv[pj] = bits2f(w.range(32 * l + 31, 32 * l));
      }
      for (int pi = 0; pi < PI; pi++) {
#pragma HLS UNROLL
        for (int pj = 0; pj < PJ; pj++) {
#pragma HLS UNROLL
          data_t p = acol.v[pi] * bv[pj];
          data_t old = (k == 0) ? (data_t)0 : acc[cur][pi][pj][s];
          acc[cur][pi][pj][s] = old + p;
        }
      }
      // drain previous block: (row pi, group s2, half h) -> 16 contiguous C values
      if (b > 0 && t < PI * S * 2) {
        int pi = t / (S * 2);
        int s2 = (t / 2) % S;
        int h = t % 2;
        cvec_t o;
        for (int u = 0; u < LANES; u++) {
#pragma HLS UNROLL
          o.v[u] = acc[prv][pi][h * LANES + u][s2];
        }
        cs.write(o);
      }
    }
  }
  int lst = (NB - 1) & 1;
  for (int t = 0; t < PI * S * 2; t++) {
#pragma HLS PIPELINE II=1
    int pi = t / (S * 2);
    int s2 = (t / 2) % S;
    int h = t % 2;
    cvec_t o;
    for (int u = 0; u < LANES; u++) {
#pragma HLS UNROLL
      o.v[u] = acc[lst][pi][h * LANES + u][s2];
    }
    cs.write(o);
  }
}

static void store_C(hls::stream<cvec_t> &cs, data_t C[I][J]) {
  for (int b = 0; b < NB; b++)
    for (int r = 0; r < PI; r++)
      for (int v = 0; v < J / LANES; v++) {
#pragma HLS PIPELINE II=1
        cvec_t o = cs.read();
        int row = b * PI + r;
        if (row < I) {
          for (int u = 0; u < LANES; u++) {
#pragma HLS UNROLL
            C[row][v * LANES + u] = o.v[u];
          }
        }
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
#pragma HLS STREAM variable=as depth=4
#pragma HLS STREAM variable=cs depth=64
  load_B(B, bs);
  load_A(A, as);
  compute(bs, as, cs);
  store_C(cs, C);
}
