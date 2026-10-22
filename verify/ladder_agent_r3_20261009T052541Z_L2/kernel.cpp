#include "kernel.h"

#if defined(__has_include)
#if __has_include(<hls_stream.h>) && __has_include(<hls_vector.h>)
#define AMM_HAVE_HLS_HEADERS 1
#endif
#endif

#ifdef AMM_HAVE_HLS_HEADERS
#include <hls_stream.h>
#include <hls_vector.h>
#else
#include <deque>
namespace hls {
template <typename T> class stream {
  std::deque<T> q_;
public:
  stream() {}
  stream(const char *) {}
  void write(const T &v) { q_.push_back(v); }
  T read() { T v = q_.front(); q_.pop_front(); return v; }
  bool empty() const { return q_.empty(); }
};
template <typename T, int N> struct vector {
  T d_[N];
  T &operator[](int i) { return d_[i]; }
  const T &operator[](int i) const { return d_[i]; }
};
}
#endif

// Output-stationary GEMM:  C[i][j] += sum_k A[i][k] * B[j][k]
//   PE array PI x PJ = 32 x 32 = 1024 fp32 MACs/cycle, k accumulated in order.
//   Output tile TI x TJ = 64 x 256 -> NSUB = 16 points time-multiplexed per PE.
//   B strip resident on chip (ping-pong), A streamed transposed, C drained
//   while next tile computes.

#define PI 32
#define PJ 32
#define KL 16
#define TI 64
#define TJ 256
#define NIB (TI / PI)
#define NJB (TJ / PJ)
#define NSUB (NIB * NJB)
#define KB (K / KL)
#define NIT (I / TI)
#define NJS (J / TJ)
#define DRAIN (TI * (TJ / KL))
#define BSTEP ((NIT * K * NSUB) / (TJ * KB))

typedef hls::vector<float, 16> vec16;
typedef hls::vector<float, PI> vecA;

static void read_A(const data_t A[I][K], hls::stream<vecA> &sA) {
  static float lbuf[PI][KL][NIB * KB];
#pragma HLS ARRAY_PARTITION variable=lbuf complete dim=1
#pragma HLS ARRAY_PARTITION variable=lbuf complete dim=2
rA_strip:
  for (int js = 0; js < NJS; js++) {
  rA_tile:
    for (int it = 0; it < NIT; it++) {
    rA_load:
      for (int n = 0; n < TI * KB; n++) {
#pragma HLS PIPELINE II=1
        int r = n / KB;
        int kb = n % KB;
        int ib = r / PI;
        int pi = r % PI;
        int row = it * TI + r;
      rA_lane:
        for (int l = 0; l < KL; l++) {
#pragma HLS UNROLL
          lbuf[pi][l][ib * KB + kb] = A[row][kb * KL + l];
        }
      }
    rA_emit:
      for (int n = 0; n < K * NIB; n++) {
#pragma HLS PIPELINE II=1
        int k = n / NIB;
        int ib = n % NIB;
        vecA v;
      rA_emit_lane:
        for (int p = 0; p < PI; p++) {
#pragma HLS UNROLL
          v[p] = lbuf[p][k % KL][ib * KB + k / KL];
        }
        sA.write(v);
      }
    }
  }
}

static void read_B(const data_t B[J][K], hls::stream<vec16> &sB) {
rB_strip:
  for (int js = 0; js < NJS; js++) {
  rB_beat:
    for (int n = 0; n < TJ * KB; n++) {
#pragma HLS PIPELINE II=1
      int j = n / KB;
      int kb = n % KB;
      vec16 v;
    rB_lane:
      for (int l = 0; l < KL; l++) {
#pragma HLS UNROLL
        v[l] = B[js * TJ + j][kb * KL + l];
      }
      sB.write(v);
    }
  }
}

static void drain_beat(float obuf[PI][PJ][NSUB], int d, hls::stream<vec16> &sO) {
#pragma HLS INLINE
  int r = d / (TJ / KL);
  int jc = d % (TJ / KL);
  int pi = r % PI;
  int ib = r / PI;
  int half = jc & 1;
  int jb = jc >> 1;
  vec16 o;
drain_lane:
  for (int l = 0; l < KL; l++) {
#pragma HLS UNROLL
    o[l] = obuf[pi][half * KL + l][ib * NJB + jb];
  }
  sO.write(o);
}

static void compute(hls::stream<vecA> &sA, hls::stream<vec16> &sB,
                    hls::stream<vec16> &sO) {
  static float bbuf[2][PJ][KL][NJB * KB];
#pragma HLS ARRAY_PARTITION variable=bbuf complete dim=1
#pragma HLS ARRAY_PARTITION variable=bbuf complete dim=2
#pragma HLS ARRAY_PARTITION variable=bbuf complete dim=3
  static float acc[PI][PJ][NSUB];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc complete dim=2
#pragma HLS BIND_STORAGE variable=acc type=ram_s2p impl=lutram
  static float obuf[PI][PJ][NSUB];
#pragma HLS ARRAY_PARTITION variable=obuf complete dim=1
#pragma HLS ARRAY_PARTITION variable=obuf complete dim=2
#pragma HLS BIND_STORAGE variable=obuf type=ram_s2p impl=lutram
  float a_reg[PI];
#pragma HLS ARRAY_PARTITION variable=a_reg complete

c_loadB0:
  for (int n = 0; n < TJ * KB; n++) {
#pragma HLS PIPELINE II=1
    int j = n / KB;
    int kb = n % KB;
    vec16 v = sB.read();
  c_loadB0_lane:
    for (int l = 0; l < KL; l++) {
#pragma HLS UNROLL
      bbuf[0][j % PJ][l][(j / PJ) * KB + kb] = v[l];
    }
  }
c_strip:
  for (int js = 0; js < NJS; js++) {
    int cur = js & 1;
    int nxt = cur ^ 1;
  c_main:
    for (int t = 0; t < NIT * K * NSUB; t++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=acc type=inter dependent=false
#pragma HLS DEPENDENCE variable=obuf type=inter dependent=false
#pragma HLS DEPENDENCE variable=bbuf type=inter dependent=false
#pragma HLS DEPENDENCE variable=bbuf type=intra dependent=false
      if (js < NJS - 1 && (t % BSTEP) == 0) {
        int nb = t / BSTEP;
        int j = nb / KB;
        int kb = nb % KB;
        vec16 v = sB.read();
      c_loadB_lane:
        for (int l = 0; l < KL; l++) {
#pragma HLS UNROLL
          bbuf[nxt][j % PJ][l][(j / PJ) * KB + kb] = v[l];
        }
      }
      int jb = t % NJB;
      int ib = (t / NJB) % NIB;
      int k = (t / NSUB) % K;
      int it = t / (NSUB * K);
      int sub = ib * NJB + jb;
      int tin = t % (K * NSUB);

      if (tin < DRAIN && (js > 0 || it > 0)) {
        drain_beat(obuf, tin, sO);
      }

      if (jb == 0) {
        vecA av = sA.read();
      c_areg:
        for (int p = 0; p < PI; p++) {
#pragma HLS UNROLL
          a_reg[p] = av[p];
        }
      }

      float bv[PJ];
#pragma HLS ARRAY_PARTITION variable=bv complete
    c_bread:
      for (int q = 0; q < PJ; q++) {
#pragma HLS UNROLL
        bv[q] = bbuf[cur][q][k % KL][jb * KB + k / KL];
      }

    c_pe_i:
      for (int p = 0; p < PI; p++) {
#pragma HLS UNROLL
      c_pe_j:
        for (int q = 0; q < PJ; q++) {
#pragma HLS UNROLL
          float old = (k == 0) ? 0.0f : acc[p][q][sub];
          float nv = old + a_reg[p] * bv[q];
          acc[p][q][sub] = nv;
          if (k == K - 1) obuf[p][q][sub] = nv;
        }
      }
    }
  }
c_final_drain:
  for (int d = 0; d < DRAIN; d++) {
#pragma HLS PIPELINE II=1
    drain_beat(obuf, d, sO);
  }
}

static void store_C(data_t C[I][J], hls::stream<vec16> &sO) {
  static float cbuf[DRAIN][KL];
#pragma HLS ARRAY_PARTITION variable=cbuf complete dim=2
s_strip:
  for (int js = 0; js < NJS; js++) {
  s_tile:
    for (int it = 0; it < NIT; it++) {
    s_loadC:
      for (int n = 0; n < DRAIN; n++) {
#pragma HLS PIPELINE II=1
        int r = n / (TJ / KL);
        int jc = n % (TJ / KL);
      s_loadC_lane:
        for (int l = 0; l < KL; l++) {
#pragma HLS UNROLL
          cbuf[n][l] = C[it * TI + r][js * TJ + jc * KL + l];
        }
      }
    s_storeC:
      for (int n = 0; n < DRAIN; n++) {
#pragma HLS PIPELINE II=1
        int r = n / (TJ / KL);
        int jc = n % (TJ / KL);
        vec16 o = sO.read();
      s_storeC_lane:
        for (int l = 0; l < KL; l++) {
#pragma HLS UNROLL
          C[it * TI + r][js * TJ + jc * KL + l] = cbuf[n][l] + o[l];
        }
      }
    }
  }
}

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS DATAFLOW
  hls::stream<vecA> sA("sA");
  hls::stream<vec16> sB("sB");
  hls::stream<vec16> sO("sO");
#pragma HLS STREAM variable=sA depth=512
#pragma HLS STREAM variable=sB depth=64
#pragma HLS STREAM variable=sO depth=64

  read_A(A, sA);
  read_B(B, sB);
  compute(sA, sB, sO);
  store_C(C, sO);
}