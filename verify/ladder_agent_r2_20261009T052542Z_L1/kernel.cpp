#include "kernel.h"

#if defined(__has_include)
#if __has_include(<hls_stream.h>) && __has_include(<hls_vector.h>)
#define AMM_HAVE_HLS_HDRS 1
#endif
#endif

#ifdef AMM_HAVE_HLS_HDRS
#include <hls_stream.h>
#include <hls_vector.h>
#else
// Software fallback for plain C++ compilers without the Vitis HLS headers.
#include <deque>
namespace hls {
template <typename T, int DEPTH = 0>
class stream {
  std::deque<T> q_;
public:
  void write(const T &v) { q_.push_back(v); }
  T read() { T v = q_.front(); q_.pop_front(); return v; }
  bool empty() const { return q_.empty(); }
};
template <typename T, int N>
struct vector {
  T d_[N];
  T &operator[](int i) { return d_[i]; }
  const T &operator[](int i) const { return d_[i]; }
};
}
#endif

// C[i][j] += sum_k A[i][k] * B[j][k]
// Output-stationary 64x64 C tiles, A row-panel (64 x K) resident and
// double-buffered, B column-panel (64 x K) double-buffered.
// Datapath: 16 (rows) x 16 (cols) x 2 (k lanes) = 512 fp32 MACs per cycle.

#define LANES 16
#define TI 64
#define TJ 64
#define PI 16
#define PJ 16
#define NBI (TI / PI)          // 4
#define NBJ (TJ / PJ)          // 4
#define NBLK (NBI * NBJ)       // 16
#define KH (K / 2)             // 512
#define KB (K / LANES)         // 64 beats per row
#define NTI (I / TI)           // 16
#define NTJ (J / TJ)           // 16
#define NTILES (NTI * NTJ)     // 256
#define PANEL_BEATS (TI * KB)  // 4096
#define CBEATS (TI * TJ / LANES) // 256
#define A_PER_TILE (PANEL_BEATS / NTJ) // 256

typedef hls::vector<float, LANES> vec_t;

static void load_A(data_t A[I][K], hls::stream<vec_t, 64> &sA) {
load_A_panel:
  for (int it = 0; it < NTI; it++) {
  load_A_beat:
    for (int d = 0; d < PANEL_BEATS; d++) {
#pragma HLS PIPELINE II=1
      int r = d / KB;
      int kb = d % KB;
      vec_t v;
    load_A_lane:
      for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
        v[l] = A[it * TI + r][kb * LANES + l];
      }
      sA.write(v);
    }
  }
}

static void load_B(data_t B[J][K], hls::stream<vec_t, 64> &sB) {
load_B_tile:
  for (int n = 0; n < NTILES; n++) {
    int jt = n % NTJ;
  load_B_beat:
    for (int d = 0; d < PANEL_BEATS; d++) {
#pragma HLS PIPELINE II=1
      int r = d / KB;
      int kb = d % KB;
      vec_t v;
    load_B_lane:
      for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
        v[l] = B[jt * TJ + r][kb * LANES + l];
      }
      sB.write(v);
    }
  }
}

static void compute(hls::stream<vec_t, 64> &sA, hls::stream<vec_t, 64> &sB,
                    hls::stream<vec_t, 256> &sAcc) {
  // [row-in-block][buf*NBI + block][k/2], even and odd k split
  static float A_e[PI][2 * NBI][KH];
  static float A_o[PI][2 * NBI][KH];
  static float B_e[PJ][2 * NBJ][KH];
  static float B_o[PJ][2 * NBJ][KH];
#pragma HLS ARRAY_PARTITION variable=A_e complete dim=1
#pragma HLS ARRAY_PARTITION variable=A_e cyclic factor=8 dim=3
#pragma HLS ARRAY_PARTITION variable=A_o complete dim=1
#pragma HLS ARRAY_PARTITION variable=A_o cyclic factor=8 dim=3
#pragma HLS ARRAY_PARTITION variable=B_e complete dim=1
#pragma HLS ARRAY_PARTITION variable=B_e cyclic factor=8 dim=3
#pragma HLS ARRAY_PARTITION variable=B_o complete dim=1
#pragma HLS ARRAY_PARTITION variable=B_o cyclic factor=8 dim=3
#pragma HLS BIND_STORAGE variable=A_e type=ram_t2p impl=bram
#pragma HLS BIND_STORAGE variable=A_o type=ram_t2p impl=bram
#pragma HLS BIND_STORAGE variable=B_e type=ram_t2p impl=bram
#pragma HLS BIND_STORAGE variable=B_o type=ram_t2p impl=bram
  static float acc[NBLK][PI][PJ];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=2
#pragma HLS ARRAY_PARTITION variable=acc complete dim=3

prologue_load:
  for (int d = 0; d < PANEL_BEATS; d++) {
#pragma HLS PIPELINE II=1
    vec_t va = sA.read();
    vec_t vb = sB.read();
    int r = d / KB;
    int kb = d % KB;
  prologue_lane:
    for (int h = 0; h < LANES / 2; h++) {
#pragma HLS UNROLL
      A_e[r % PI][r / PI][kb * 8 + h] = va[2 * h];
      A_o[r % PI][r / PI][kb * 8 + h] = va[2 * h + 1];
      B_e[r % PJ][r / PJ][kb * 8 + h] = vb[2 * h];
      B_o[r % PJ][r / PJ][kb * 8 + h] = vb[2 * h + 1];
    }
  }

tile_loop:
  for (int n = 0; n < NTILES; n++) {
    int it = n / NTJ;
    int jt = n % NTJ;
    int abuf = it & 1;
    int bbuf = n & 1;
    bool more_b = (n < NTILES - 1);
    bool more_a = (it < NTI - 1);

  mac_loop:
    for (int t = 0; t < KH * NBLK; t++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=acc inter false
#pragma HLS DEPENDENCE variable=A_e inter false
#pragma HLS DEPENDENCE variable=A_o inter false
#pragma HLS DEPENDENCE variable=B_e inter false
#pragma HLS DEPENDENCE variable=B_o inter false
#pragma HLS DEPENDENCE variable=A_e intra false
#pragma HLS DEPENDENCE variable=A_o intra false
#pragma HLS DEPENDENCE variable=B_e intra false
#pragma HLS DEPENDENCE variable=B_o intra false
      int k2 = t / NBLK;
      int blk = t % NBLK;
      int bi = blk / NBJ;
      int bj = blk % NBJ;

      // prefetch next B tile
      if (more_b && t < PANEL_BEATS) {
        vec_t vb = sB.read();
        int r = t / KB;
        int kb = t % KB;
      pf_b_lane:
        for (int h = 0; h < LANES / 2; h++) {
#pragma HLS UNROLL
          B_e[r % PJ][(1 - bbuf) * NBJ + r / PJ][kb * 8 + h] = vb[2 * h];
          B_o[r % PJ][(1 - bbuf) * NBJ + r / PJ][kb * 8 + h] = vb[2 * h + 1];
        }
      }
      // prefetch slice of next A panel
      if (more_a && t < A_PER_TILE) {
        vec_t va = sA.read();
        int idx = jt * A_PER_TILE + t;
        int r = idx / KB;
        int kb = idx % KB;
      pf_a_lane:
        for (int h = 0; h < LANES / 2; h++) {
#pragma HLS UNROLL
          A_e[r % PI][(1 - abuf) * NBI + r / PI][kb * 8 + h] = va[2 * h];
          A_o[r % PI][(1 - abuf) * NBI + r / PI][kb * 8 + h] = va[2 * h + 1];
        }
      }

      float a0[PI], a1[PI], b0[PJ], b1[PJ];
#pragma HLS ARRAY_PARTITION variable=a0 complete
#pragma HLS ARRAY_PARTITION variable=a1 complete
#pragma HLS ARRAY_PARTITION variable=b0 complete
#pragma HLS ARRAY_PARTITION variable=b1 complete
    rd_a:
      for (int p = 0; p < PI; p++) {
#pragma HLS UNROLL
        a0[p] = A_e[p][abuf * NBI + bi][k2];
        a1[p] = A_o[p][abuf * NBI + bi][k2];
      }
    rd_b:
      for (int q = 0; q < PJ; q++) {
#pragma HLS UNROLL
        b0[q] = B_e[q][bbuf * NBJ + bj][k2];
        b1[q] = B_o[q][bbuf * NBJ + bj][k2];
      }
    pe_row:
      for (int p = 0; p < PI; p++) {
#pragma HLS UNROLL
      pe_col:
        for (int q = 0; q < PJ; q++) {
#pragma HLS UNROLL
          float m0 = a0[p] * b0[q];
          float m1 = a1[p] * b1[q];
          float s = m0 + m1;
#pragma HLS BIND_OP variable=s op=fadd impl=fabric
          float prev = (k2 == 0) ? 0.0f : acc[blk][p][q];
          acc[blk][p][q] = prev + s;
        }
      }
    }

  drain_loop:
    for (int d = 0; d < CBEATS; d++) {
#pragma HLS PIPELINE II=1
      int r = d / NBJ;
      int cb = d % NBJ;
      vec_t v;
    drain_lane:
      for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
        v[l] = acc[(r / PI) * NBJ + cb][r % PI][l];
      }
      sAcc.write(v);
    }
  }
}

static void store_C(data_t C[I][J], hls::stream<vec_t, 256> &sAcc) {
  vec_t cbuf[CBEATS];
store_tile:
  for (int n = 0; n < NTILES; n++) {
    int it = n / NTJ;
    int jt = n % NTJ;
  rd_c:
    for (int d = 0; d < CBEATS; d++) {
#pragma HLS PIPELINE II=1
      int r = d / NBJ;
      int cb = d % NBJ;
      vec_t v;
    rd_c_lane:
      for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
        v[l] = C[it * TI + r][jt * TJ + cb * LANES + l];
      }
      cbuf[d] = v;
    }
  wr_c:
    for (int d = 0; d < CBEATS; d++) {
#pragma HLS PIPELINE II=1
      int r = d / NBJ;
      int cb = d % NBJ;
      vec_t s = sAcc.read();
      vec_t c = cbuf[d];
    wr_c_lane:
      for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
        C[it * TI + r][jt * TJ + cb * LANES + l] = c[l] + s[l];
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
  hls::stream<vec_t, 64> sA;
  hls::stream<vec_t, 64> sB;
  hls::stream<vec_t, 256> sAcc;

  load_A(A, sA);
  load_B(B, sB);
  compute(sA, sB, sAcc);
  store_C(C, sAcc);
}