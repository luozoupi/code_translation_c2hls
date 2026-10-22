#include "kernel.h"
#include <hls_stream.h>
#include <hls_vector.h>

// Output-stationary lane array: TI x TJ fp32 MAC lanes, each lane owns L interleaved outputs
// (hides the fadd latency) and accumulates over k in the exact sequential order of plain.cpp.
//   row i = ib*TI + ii,  col j = jj*TJ + jx   (TJ*L == J)
// Phase 1 (i-blocks 0..G-1): k-chunk-outer order while B streams in chunk by chunk (hides the B load).
// Phase 2 (i-blocks G..NIB-1): i-block-outer order from the fully on-chip B.
// Results are drained as sequential 512-bit beats of C while later i-blocks compute.
#define TI 32
#define TJ 64
#define L (J / TJ)          // 16
#define NIB (I / TI)        // 32
#define W 16                // floats per 512-bit beat
#define KW (K / W)          // 64 beats per row
#define G 4                 // i-blocks in phase 1
#define KC 16               // phase-1 k chunk
#define NKC (K / KC)        // 16
#define CW (KC / W)         // 4 beats per row per chunk
#define P1 (NKC * G * KC * L)          // phase-1 iterations (65536)
#define P2B (K * L)                    // phase-2 iterations per i-block (16384)
#define NBEAT (I * J / W)              // C beats (65536)
#define BPB (TI * J / W)               // C beats per i-block (2048)
#define LAG 64

typedef hls::vector<float, TI> apkt_t;
typedef hls::vector<float, W> beat_t;

static void loadB(data_t B[J][K], hls::stream<beat_t> &bs) {
  for (int kc = 0; kc < NKC; kc++)
    for (int j = 0; j < J; j++)
      for (int w = 0; w < CW; w++) {
#pragma HLS PIPELINE II=1
        beat_t v;
        for (int u = 0; u < W; u++) v[u] = B[j][kc * KC + w * W + u];
        bs.write(v);
      }
}

static void readA(data_t A[I][K], hls::stream<apkt_t> &as) {
  float abuf[TI][K];
#pragma HLS ARRAY_PARTITION variable=abuf dim=1 complete
#pragma HLS ARRAY_PARTITION variable=abuf dim=2 cyclic factor=16
  // phase 1: per (chunk, i-block) TI rows x KC
  for (int kc = 0; kc < NKC; kc++)
    for (int ib = 0; ib < G; ib++) {
      for (int r = 0; r < TI; r++)
        for (int w = 0; w < CW; w++) {
#pragma HLS PIPELINE II=1
          for (int u = 0; u < W; u++) abuf[r][w * W + u] = A[ib * TI + r][kc * KC + w * W + u];
        }
      for (int k = 0; k < KC; k++) {
#pragma HLS PIPELINE II=1
        apkt_t p;
        for (int r = 0; r < TI; r++) p[r] = abuf[r][k];
        as.write(p);
      }
    }
  // phase 2: per i-block TI rows x K
  for (int ib = G; ib < NIB; ib++) {
    for (int r = 0; r < TI; r++)
      for (int w = 0; w < KW; w++) {
#pragma HLS PIPELINE II=1
        for (int u = 0; u < W; u++) abuf[r][w * W + u] = A[ib * TI + r][w * W + u];
      }
    for (int k = 0; k < K; k++) {
#pragma HLS PIPELINE II=1
      apkt_t p;
      for (int r = 0; r < TI; r++) p[r] = abuf[r][k];
      as.write(p);
    }
  }
}

// iteration (global count) at which i-block d has written its final results
static int final_iter(int d) {
#pragma HLS INLINE
  return d < G ? (NKC - 1) * G * KC * L + d * KC * L + KC * L - 1 : P1 + (d - G) * P2B + P2B - 1;
}

static void compute(hls::stream<beat_t> &bs, hls::stream<apkt_t> &as, hls::stream<beat_t> &cs) {
  // Bbuf[jx][u][jj*KW + w] = B[jj*TJ + jx][w*W + u]
  float Bbuf[TJ][W][L * KW];
#pragma HLS ARRAY_PARTITION variable=Bbuf dim=1 complete
#pragma HLS ARRAY_PARTITION variable=Bbuf dim=2 complete
#pragma HLS BIND_STORAGE variable=Bbuf type=ram_s2p impl=bram
  float acc[TI][TJ][G * L];
#pragma HLS ARRAY_PARTITION variable=acc dim=1 complete
#pragma HLS ARRAY_PARTITION variable=acc dim=2 complete
#pragma HLS BIND_STORAGE variable=acc type=ram_s2p impl=lutram
  float obuf[TI][TJ][G * L];   // final results of i-block d at [(d % G)*L + jj]
#pragma HLS ARRAY_PARTITION variable=obuf dim=1 complete
#pragma HLS ARRAY_PARTITION variable=obuf dim=2 complete
#pragma HLS BIND_STORAGE variable=obuf type=ram_s2p impl=lutram

  // B chunk 0
  for (int t = 0; t < J * CW; t++) {
#pragma HLS PIPELINE II=1
    int j = t / CW, w = t % CW;
    beat_t v = bs.read();
    for (int u = 0; u < W; u++) Bbuf[j % TJ][u][(j / TJ) * KW + w] = v[u];
  }

  int dc = 0;   // C beats drained so far
  apkt_t a;

  // phase 1
  for (int n = 0; n < P1; n++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=acc type=inter direction=RAW distance=16 dependent=true
#pragma HLS DEPENDENCE variable=obuf type=inter dependent=false
#pragma HLS DEPENDENCE variable=Bbuf type=inter dependent=false
    int jj = n % L, k = (n / L) % KC, ib = (n / (KC * L)) % G, kc = n / (G * KC * L);
    int kg = kc * KC + k;
    if (jj == 0) a = as.read();

    // stream in the next B chunk, one beat per iteration
    if (kc < NKC - 1) {
      int t = n % (G * KC * L), j = t / CW, w = (kc + 1) * CW + t % CW;
      beat_t v = bs.read();
      for (int u = 0; u < W; u++) Bbuf[j % TJ][u][(j / TJ) * KW + w] = v[u];
    }

    // drain finished i-blocks
    int d = dc / BPB;
    if (d < NIB && n >= final_iter(d) + LAG) {
      int rem = dc % BPB, ii = rem / KW, w = rem % KW;
      beat_t c;
      for (int u = 0; u < W; u++) c[u] = obuf[ii][(w % 4) * W + u][(d % G) * L + w / 4];
      cs.write(c);
      dc++;
    }

    float b[TJ];
#pragma HLS ARRAY_PARTITION variable=b complete
    for (int jx = 0; jx < TJ; jx++) b[jx] = Bbuf[jx][kg % W][jj * KW + kg / W];
    for (int ii = 0; ii < TI; ii++)
      for (int jx = 0; jx < TJ; jx++) {
        float p = a[ii] * b[jx];
#pragma HLS BIND_OP variable=p op=fmul impl=fulldsp
        float o = (kg == 0) ? 0.0f : acc[ii][jx][ib * L + jj];
        float s = o + p;
#pragma HLS BIND_OP variable=s op=fadd impl=fulldsp
        acc[ii][jx][ib * L + jj] = s;
        if (kg == K - 1) obuf[ii][jx][ib * L + jj] = s;
      }
  }

  // phase 2
  for (int n = 0; n < (NIB - G) * P2B; n++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=acc type=inter direction=RAW distance=16 dependent=true
#pragma HLS DEPENDENCE variable=obuf type=inter dependent=false
    int jj = n % L, k = (n / L) % K, ib = G + n / P2B;
    if (jj == 0) a = as.read();

    int d = dc / BPB;
    if (d < NIB && P1 + n >= final_iter(d) + LAG) {
      int rem = dc % BPB, ii = rem / KW, w = rem % KW;
      beat_t c;
      for (int u = 0; u < W; u++) c[u] = obuf[ii][(w % 4) * W + u][(d % G) * L + w / 4];
      cs.write(c);
      dc++;
    }

    float b[TJ];
#pragma HLS ARRAY_PARTITION variable=b complete
    for (int jx = 0; jx < TJ; jx++) b[jx] = Bbuf[jx][k % W][jj * KW + k / W];
    for (int ii = 0; ii < TI; ii++)
      for (int jx = 0; jx < TJ; jx++) {
        float p = a[ii] * b[jx];
#pragma HLS BIND_OP variable=p op=fmul impl=fulldsp
        float o = (k == 0) ? 0.0f : acc[ii][jx][jj];
        float s = o + p;
#pragma HLS BIND_OP variable=s op=fadd impl=fulldsp
        acc[ii][jx][jj] = s;
        if (k == K - 1) obuf[ii][jx][(ib % G) * L + jj] = s;
      }
  }

  // drain the rest
  for (; dc < NBEAT; dc++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=2048 max=2048
    int d = dc / BPB, rem = dc % BPB, ii = rem / KW, w = rem % KW;
    beat_t c;
    for (int u = 0; u < W; u++) c[u] = obuf[ii][(w % 4) * W + u][(d % G) * L + w / 4];
    cs.write(c);
  }
}

static void writeC(hls::stream<beat_t> &cs, data_t C[I][J]) {
  for (int i = 0; i < I; i++)
    for (int w = 0; w < J / W; w++) {
#pragma HLS PIPELINE II=1
      beat_t c = cs.read();
      for (int u = 0; u < W; u++) C[i][w * W + u] = c[u];
    }
}

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmemA depth=1048576
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmemB depth=1048576
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmemC depth=1048576
#pragma HLS DATAFLOW
  hls::stream<beat_t> bs;
#pragma HLS STREAM variable=bs depth=64
  hls::stream<apkt_t> as;
#pragma HLS STREAM variable=as depth=256
  hls::stream<beat_t> cs;
#pragma HLS STREAM variable=cs depth=64
  loadB(B, bs);
  readA(A, as);
  compute(bs, as, cs);
  writeC(cs, C);
}
