#include "kernel.h"
#include <hls_stream.h>

// C[i][j] = sum_k A[i][k] * B[j][k]
// k is accumulated strictly in order (bit-exact with the sequential reference).
//
// Architecture (DATAFLOW, stream connected):
//   readA  : A (m_axi gmem0) -> sA  (4 floats / beat)
//   readB  : B (m_axi gmem1) -> sB  (8 floats / beat)
//   compute: output-stationary TI x TJ tile, TI*UJ = 512 MACs/cycle.
//            Ping-pong A/B/C buffers; prefetch of the next A/B tile and drain of
//            the previous C tile are fused into the compute pipeline (full overlap).
//   writeC : sC -> C (m_axi gmem2)

#define TI 64               // A rows per tile (fully unrolled)
#define TJ 128              // B rows per tile
#define UJ 8                // j lanes per cycle
#define JG (TJ / UJ)        // 16: recurrence distance on C tile
#define NIT (I / TI)        // 16
#define NJT (J / TJ)        // 8
#define NT (NIT * NJT)      // 128 tiles
#define NITER (K * JG)      // 16384 compute iterations per tile
#define WA 4                // A floats per stream beat
#define WB 8                // B floats per stream beat
#define WC 8                // C floats per stream beat
#define A_BEATS (TI * K / WA)   // 16384
#define B_BEATS (TJ * K / WB)   // 16384
#define C_BEATS (TI * TJ / WC)  // 1024
#define C_DRAIN_START 1024

struct va_t { data_t v[WA]; };
struct vb_t { data_t v[WB]; };
struct vc_t { data_t v[WC]; };

static void readA(data_t A[I][K], hls::stream<va_t> &sA) {
rdA_it:
  for (int it = 0; it < NIT; it++) {
  rdA_i:
    for (int i = 0; i < TI; i++) {
    rdA_k:
      for (int k = 0; k < K; k += WA) {
#pragma HLS PIPELINE II=1
        va_t x;
        for (int u = 0; u < WA; u++) {
#pragma HLS UNROLL
          x.v[u] = A[it * TI + i][k + u];
        }
        sA.write(x);
      }
    }
  }
}

static void readB(data_t B[J][K], hls::stream<vb_t> &sB) {
rdB_it:
  for (int it = 0; it < NIT; it++) {
  rdB_jt:
    for (int jt = 0; jt < NJT; jt++) {
    rdB_j:
      for (int j = 0; j < TJ; j++) {
      rdB_k:
        for (int k = 0; k < K; k += WB) {
#pragma HLS PIPELINE II=1
          vb_t x;
          for (int u = 0; u < WB; u++) {
#pragma HLS UNROLL
            x.v[u] = B[jt * TJ + j][k + u];
          }
          sB.write(x);
        }
      }
    }
  }
}

static void writeC(hls::stream<vc_t> &sC, data_t C[I][J]) {
wrC_it:
  for (int it = 0; it < NIT; it++) {
  wrC_jt:
    for (int jt = 0; jt < NJT; jt++) {
    wrC_i:
      for (int i = 0; i < TI; i++) {
      wrC_j:
        for (int j = 0; j < TJ; j += WC) {
#pragma HLS PIPELINE II=1
          vc_t x = sC.read();
          for (int u = 0; u < WC; u++) {
#pragma HLS UNROLL
            C[it * TI + i][jt * TJ + j + u] = x.v[u];
          }
        }
      }
    }
  }
}

static void compute(hls::stream<va_t> &sA, hls::stream<vb_t> &sB, hls::stream<vc_t> &sC) {
  data_t A0[TI][K], A1[TI][K];
#pragma HLS ARRAY_PARTITION variable=A0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=A0 cyclic factor=WA dim=2
#pragma HLS ARRAY_PARTITION variable=A1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=A1 cyclic factor=WA dim=2
  data_t B0[TJ][K], B1[TJ][K];
#pragma HLS ARRAY_PARTITION variable=B0 cyclic factor=UJ dim=1
#pragma HLS ARRAY_PARTITION variable=B0 cyclic factor=WB dim=2
#pragma HLS ARRAY_PARTITION variable=B1 cyclic factor=UJ dim=1
#pragma HLS ARRAY_PARTITION variable=B1 cyclic factor=WB dim=2
  data_t C0[TI][TJ], C1[TI][TJ];
#pragma HLS ARRAY_PARTITION variable=C0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=C0 cyclic factor=UJ dim=2
#pragma HLS ARRAY_PARTITION variable=C1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=C1 cyclic factor=UJ dim=2

  // Prologue: first A tile and first B tile.
prologue:
  for (int n = 0; n < A_BEATS; n++) {
#pragma HLS PIPELINE II=1
    va_t xa = sA.read();
    int ra = n / (K / WA);
    int ka = (n % (K / WA)) * WA;
    for (int u = 0; u < WA; u++) {
#pragma HLS UNROLL
      A0[ra][ka + u] = xa.v[u];
    }
    vb_t xb = sB.read();
    int rb = n / (K / WB);
    int kb = (n % (K / WB)) * WB;
    for (int u = 0; u < WB; u++) {
#pragma HLS UNROLL
      B0[rb][kb + u] = xb.v[u];
    }
  }

tiles:
  for (int t = 0; t < NT; t++) {
    const int it = t / NJT;
    const int jt = t % NJT;
    const bool pA = (it & 1);
    const bool pB = (t & 1);
    const bool preB = (t + 1 < NT);
    const bool preA = (jt == NJT - 1) && (it + 1 < NIT);
    const bool drain = (t > 0);
  main:
    for (int n = 0; n < NITER; n++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=C0 inter distance=16 true
#pragma HLS DEPENDENCE variable=C1 inter distance=16 true
#pragma HLS DEPENDENCE variable=A0 inter false
#pragma HLS DEPENDENCE variable=A1 inter false
#pragma HLS DEPENDENCE variable=B0 inter false
#pragma HLS DEPENDENCE variable=B1 inter false
      const int k = n / JG;
      const int jg = n % JG;

      // ---------------- compute (512 MACs) ----------------
      data_t a[TI];
#pragma HLS ARRAY_PARTITION variable=a complete
      data_t b[UJ];
#pragma HLS ARRAY_PARTITION variable=b complete
      for (int i = 0; i < TI; i++) {
#pragma HLS UNROLL
        a[i] = pA ? A1[i][k] : A0[i][k];
      }
      for (int u = 0; u < UJ; u++) {
#pragma HLS UNROLL
        b[u] = pB ? B1[jg * UJ + u][k] : B0[jg * UJ + u][k];
      }
      for (int i = 0; i < TI; i++) {
#pragma HLS UNROLL
        for (int u = 0; u < UJ; u++) {
#pragma HLS UNROLL
          const int j = jg * UJ + u;
          data_t old = pB ? C1[i][j] : C0[i][j];
          data_t prev = (k == 0) ? (data_t)0 : old;
          data_t res = prev + a[i] * b[u];
          if (pB) C1[i][j] = res;
          else    C0[i][j] = res;
        }
      }

      // ---------------- prefetch next B tile ----------------
      if (preB) {
        vb_t xb = sB.read();
        int rb = n / (K / WB);
        int kb = (n % (K / WB)) * WB;
        for (int u = 0; u < WB; u++) {
#pragma HLS UNROLL
          if (pB) B0[rb][kb + u] = xb.v[u];
          else    B1[rb][kb + u] = xb.v[u];
        }
      }

      // ---------------- prefetch next A tile ----------------
      if (preA) {
        va_t xa = sA.read();
        int ra = n / (K / WA);
        int ka = (n % (K / WA)) * WA;
        for (int u = 0; u < WA; u++) {
#pragma HLS UNROLL
          if (pA) A0[ra][ka + u] = xa.v[u];
          else    A1[ra][ka + u] = xa.v[u];
        }
      }

      // ---------------- drain previous C tile ----------------
      if (drain && n >= C_DRAIN_START && n < C_DRAIN_START + C_BEATS) {
        int m = n - C_DRAIN_START;
        int ri = m / (TJ / WC);
        int cj = (m % (TJ / WC)) * WC;
        vc_t xc;
        for (int u = 0; u < WC; u++) {
#pragma HLS UNROLL
          xc.v[u] = pB ? C0[ri][cj + u] : C1[ri][cj + u];
        }
        sC.write(xc);
      }
    }
  }

  // Epilogue: drain the last C tile (tile NT-1 lives in buffer (NT-1)&1).
epilogue:
  for (int m = 0; m < C_BEATS; m++) {
#pragma HLS PIPELINE II=1
    int ri = m / (TJ / WC);
    int cj = (m % (TJ / WC)) * WC;
    vc_t xc;
    for (int u = 0; u < WC; u++) {
#pragma HLS UNROLL
      xc.v[u] = ((NT - 1) & 1) ? C1[ri][cj + u] : C0[ri][cj + u];
    }
    sC.write(xc);
  }
}

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=1048576 max_widen_bitwidth=512 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=return

#pragma HLS DATAFLOW
  hls::stream<va_t> sA("sA");
  hls::stream<vb_t> sB("sB");
  hls::stream<vc_t> sC("sC");
#pragma HLS STREAM variable=sA depth=64
#pragma HLS STREAM variable=sB depth=64
#pragma HLS STREAM variable=sC depth=64

  readA(A, sA);
  readB(B, sB);
  compute(sA, sB, sC);
  writeC(sC, C);
}
