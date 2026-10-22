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
// Output-stationary 128x64 C tiles, A row-panel (128 x K) resident and
// double-buffered, B column-panel (64 x K) double-buffered.
// Datapath: 32 (rows) x 16 (cols) x 4 (k lanes) = 2048 fp32 MACs per cycle.
// Results are copied into a shadow register file at the last k step and
// drained during the next tile's MAC loop (drain overlapped with compute).

#define LANES 16
#define TI 128
#define TJ 64
#define PI 32
#define PJ 16
#define KL 4                   // k lanes per cycle
#define PI_DSP 16              // rows whose accumulator fadd uses DSP
#define NBI (TI / PI)          // 4
#define NBJ (TJ / PJ)          // 4
#define NBLK (NBI * NBJ)       // 16
#define KQ (K / KL)            // 256
#define KB (K / LANES)         // 64 beats per row
#define BPK (LANES / KL)       // 4 entries per lane-array per beat
#define NTI (I / TI)           // 8
#define NTJ (J / TJ)           // 16
#define NTILES (NTI * NTJ)     // 128
#define PANEL_BEATS (TJ * KB)  // 4096 (B panel)
#define A_PANEL_BEATS (TI * KB) // 8192 (A panel)
#define CBEATS (TI * TJ / LANES) // 512
#define A_PER_TILE (A_PANEL_BEATS / NTJ) // 512
#define MAC_ITERS (KQ * NBLK)  // 4096

typedef hls::vector<float, LANES> vec_t;

static void load_A(data_t A[I][K], hls::stream<vec_t, 64> &sA) {
load_A_panel:
  for (int it = 0; it < NTI; it++) {
  load_A_beat:
    for (int d = 0; d < A_PANEL_BEATS; d++) {
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
                    hls::stream<vec_t, 512> &sAcc) {
  // [k lane][row-in-block][buf*NB + block][k/KL]
  static float Aq[KL][PI][2 * NBI][KQ];
  static float Bq[KL][PJ][2 * NBJ][KQ];
#pragma HLS ARRAY_PARTITION variable=Aq complete dim=1
#pragma HLS ARRAY_PARTITION variable=Aq complete dim=2
#pragma HLS ARRAY_PARTITION variable=Aq cyclic factor=4 dim=4
#pragma HLS ARRAY_PARTITION variable=Bq complete dim=1
#pragma HLS ARRAY_PARTITION variable=Bq complete dim=2
#pragma HLS ARRAY_PARTITION variable=Bq cyclic factor=4 dim=4
#pragma HLS BIND_STORAGE variable=Aq type=ram_t2p impl=bram
#pragma HLS BIND_STORAGE variable=Bq type=ram_t2p impl=bram
  static float acc[NBLK][PI][PJ];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=2
#pragma HLS ARRAY_PARTITION variable=acc complete dim=3
  static float osh[NBLK][PI][PJ];
#pragma HLS ARRAY_PARTITION variable=osh complete dim=2
#pragma HLS ARRAY_PARTITION variable=osh complete dim=3

prologue_load:
  for (int d = 0; d < A_PANEL_BEATS; d++) {
#pragma HLS PIPELINE II=1
    vec_t va = sA.read();
    int r = d / KB;
    int kb = d % KB;
  prologue_lane_a:
    for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
      Aq[l % KL][r % PI][r / PI][kb * BPK + l / KL] = va[l];
    }
    if (d < PANEL_BEATS) {
      vec_t vb = sB.read();
    prologue_lane_b:
      for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
        Bq[l % KL][r % PJ][r / PJ][kb * BPK + l / KL] = vb[l];
      }
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
    bool drain_prev = (n > 0);

  mac_loop:
    for (int t = 0; t < MAC_ITERS; t++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=acc inter false
#pragma HLS DEPENDENCE variable=osh inter false
#pragma HLS DEPENDENCE variable=osh intra false
#pragma HLS DEPENDENCE variable=Aq inter false
#pragma HLS DEPENDENCE variable=Bq inter false
#pragma HLS DEPENDENCE variable=Aq intra false
#pragma HLS DEPENDENCE variable=Bq intra false
      int k4 = t / NBLK;
      int blk = t % NBLK;
      int bi = blk / NBJ;
      int bj = blk % NBJ;

      // drain previous tile's results (overlapped with compute)
      if (drain_prev && t < CBEATS) {
        int r = t / NBJ;
        int cb = t % NBJ;
        vec_t v;
      drain_lane:
        for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
          v[l] = osh[(r / PI) * NBJ + cb][r % PI][l];
        }
        sAcc.write(v);
      }

      // prefetch next B tile
      if (more_b && t < PANEL_BEATS) {
        vec_t vb = sB.read();
        int r = t / KB;
        int kb = t % KB;
      pf_b_lane:
        for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
          Bq[l % KL][r % PJ][(1 - bbuf) * NBJ + r / PJ][kb * BPK + l / KL] = vb[l];
        }
      }
      // prefetch slice of next A panel
      if (more_a && t < A_PER_TILE) {
        vec_t va = sA.read();
        int idx = jt * A_PER_TILE + t;
        int r = idx / KB;
        int kb = idx % KB;
      pf_a_lane:
        for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
          Aq[l % KL][r % PI][(1 - abuf) * NBI + r / PI][kb * BPK + l / KL] = va[l];
        }
      }

      float a[KL][PI], b[KL][PJ];
#pragma HLS ARRAY_PARTITION variable=a complete dim=0
#pragma HLS ARRAY_PARTITION variable=b complete dim=0
    rd_a:
      for (int p = 0; p < PI; p++) {
#pragma HLS UNROLL
      rd_a_k:
        for (int h = 0; h < KL; h++) {
#pragma HLS UNROLL
          a[h][p] = Aq[h][p][abuf * NBI + bi][k4];
        }
      }
    rd_b:
      for (int q = 0; q < PJ; q++) {
#pragma HLS UNROLL
      rd_b_k:
        for (int h = 0; h < KL; h++) {
#pragma HLS UNROLL
          b[h][q] = Bq[h][q][bbuf * NBJ + bj][k4];
        }
      }
      bool first = (k4 == 0);
      bool last = (k4 == KQ - 1);
    pe_row_dsp:
      for (int p = 0; p < PI_DSP; p++) {
#pragma HLS UNROLL
      pe_col_dsp:
        for (int q = 0; q < PJ; q++) {
#pragma HLS UNROLL
          float m0 = a[0][p] * b[0][q];
          float m1 = a[1][p] * b[1][q];
          float m2 = a[2][p] * b[2][q];
          float m3 = a[3][p] * b[3][q];
          float s01 = m0 + m1;
#pragma HLS BIND_OP variable=s01 op=fadd impl=fabric
          float s23 = m2 + m3;
#pragma HLS BIND_OP variable=s23 op=fadd impl=fabric
          float s = s01 + s23;
#pragma HLS BIND_OP variable=s op=fadd impl=fabric
          float prev = first ? 0.0f : acc[blk][p][q];
          float nv = prev + s;
#pragma HLS BIND_OP variable=nv op=fadd impl=fulldsp
          acc[blk][p][q] = nv;
          if (last) osh[blk][p][q] = nv;
        }
      }
    pe_row_fab:
      for (int p = PI_DSP; p < PI; p++) {
#pragma HLS UNROLL
      pe_col_fab:
        for (int q = 0; q < PJ; q++) {
#pragma HLS UNROLL
          float m0 = a[0][p] * b[0][q];
          float m1 = a[1][p] * b[1][q];
          float m2 = a[2][p] * b[2][q];
          float m3 = a[3][p] * b[3][q];
          float s01 = m0 + m1;
#pragma HLS BIND_OP variable=s01 op=fadd impl=fabric
          float s23 = m2 + m3;
#pragma HLS BIND_OP variable=s23 op=fadd impl=fabric
          float s = s01 + s23;
#pragma HLS BIND_OP variable=s op=fadd impl=fabric
          float prev = first ? 0.0f : acc[blk][p][q];
          float nv = prev + s;
#pragma HLS BIND_OP variable=nv op=fadd impl=fabric
          acc[blk][p][q] = nv;
          if (last) osh[blk][p][q] = nv;
        }
      }
    }
  }

epilogue_drain:
  for (int d = 0; d < CBEATS; d++) {
#pragma HLS PIPELINE II=1
    int r = d / NBJ;
    int cb = d % NBJ;
    vec_t v;
  epi_drain_lane:
    for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
      v[l] = osh[(r / PI) * NBJ + cb][r % PI][l];
    }
    sAcc.write(v);
  }
}

static void store_C(data_t C[I][J], hls::stream<vec_t, 512> &sAcc) {
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
        v[l] = (data_t)0;
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
  hls::stream<vec_t, 512> sAcc;

  load_A(A, sA);
  load_B(B, sB);
  compute(sA, sB, sAcc);
  store_C(C, sAcc);
}