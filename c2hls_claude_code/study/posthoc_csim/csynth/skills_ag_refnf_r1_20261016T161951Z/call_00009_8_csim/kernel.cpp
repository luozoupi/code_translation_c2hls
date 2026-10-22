#include "kernel.h"
#include <hls_stream.h>
#include <hls_vector.h>

// C[i][j] = sum_k A[i][k] * B[j][k]  (serial k order per output, bit-matching the reference)
//
// Architecture: PI x PJ fp32 MACs per cycle.
//  - A B-panel of PJ rows (all K) stays resident on chip.
//  - A tiles of G*PI rows stream in (ping-pong buffered, overlapped with compute).
//  - Each cycle: one k, one row-group g; PI A values x PJ B values -> PI*PJ MACs.
//  - Row-groups g are interleaved (G = 8) so the fp-add recurrence on each
//    accumulator has distance 8 cycles -> II = 1 with exact serial k order.

#define LANES 16
#define PI 16
#define PJ 48
#define G 8
#define KB (K / LANES)            // 64 beats per row
#define TROWS (G * PI)            // 128 rows of A per tile
#define TPP (I / TROWS)           // tiles per B panel = 8
#define NPAN ((J + PJ - 1) / PJ)  // B panels (last one zero-padded)
#define NT (TPP * NPAN)           // tiles total
#define STEPS (G * K)             // 8192 cycles per tile
#define OUTBEATS (TROWS * (PJ / LANES))  // 384 beats of C per tile

typedef hls::vector<float, LANES> vec_t;

static void load_A(data_t A[I][K], hls::stream<vec_t> &sA) {
load_A_pan:
  for (int p = 0; p < NPAN; p++) {
  load_A_row:
    for (int r = 0; r < I; r++) {
    load_A_beat:
      for (int kb = 0; kb < KB; kb++) {
#pragma HLS PIPELINE II=1
        vec_t v;
      load_A_lane:
        for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
          v[l] = A[r][kb * LANES + l];
        }
        sA.write(v);
      }
    }
  }
}

static void load_B(data_t B[J][K], hls::stream<vec_t> &sB) {
load_B_row:
  for (int r = 0; r < J; r++) {
  load_B_beat:
    for (int kb = 0; kb < KB; kb++) {
#pragma HLS PIPELINE II=1
      vec_t v;
    load_B_lane:
      for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
        v[l] = B[r][kb * LANES + l];
      }
      sB.write(v);
    }
  }
  // zero rows padding the last B panel
load_B_pad:
  for (int t = 0; t < (NPAN * PJ - J) * KB; t++) {
#pragma HLS PIPELINE II=1
    vec_t z;
  load_B_zero:
    for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
      z[l] = 0.0f;
    }
    sB.write(z);
  }
}

static void compute(hls::stream<vec_t> &sA, hls::stream<vec_t> &sB,
                    hls::stream<vec_t> &sC) {
  static vec_t Abuf[2][PI][G * KB];
#pragma HLS ARRAY_PARTITION variable=Abuf complete dim=1
#pragma HLS ARRAY_PARTITION variable=Abuf complete dim=2
  static vec_t Bres[PJ][KB];
#pragma HLS ARRAY_PARTITION variable=Bres complete dim=1
#pragma HLS BIND_STORAGE variable=Bres type=ram_s2p impl=lutram
  static float acc[G][PI][PJ];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=2
#pragma HLS ARRAY_PARTITION variable=acc complete dim=3
#pragma HLS BIND_STORAGE variable=acc type=ram_s2p impl=lutram
  static float Cbuf[G][PI][PJ];
#pragma HLS ARRAY_PARTITION variable=Cbuf complete dim=2
#pragma HLS ARRAY_PARTITION variable=Cbuf complete dim=3
#pragma HLS BIND_STORAGE variable=Cbuf type=ram_s2p impl=lutram

tile_loop:
  for (int n = 0; n <= NT; n++) {
    if (n >= 1 && ((n - 1) % TPP) == 0) {
    load_B_panel:
      for (int t = 0; t < PJ * KB; t++) {
#pragma HLS PIPELINE II=1
        Bres[t / KB][t % KB] = sB.read();
      }
    }
    const int wp = n & 1;
    const int rp = (n + 1) & 1;
  step_loop:
    for (int t = 0; t < STEPS; t++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=Abuf inter false
#pragma HLS DEPENDENCE variable=Abuf intra false
#pragma HLS DEPENDENCE variable=Cbuf inter false
#pragma HLS DEPENDENCE variable=Cbuf intra false
#pragma HLS DEPENDENCE variable=acc type=inter direction=RAW distance=8 dependent=true
      // ---- load next A tile (ping-pong) ----
      if (n < NT) {
        vec_t v = sA.read();
        int r = t / KB;
        int kb = t % KB;
        int lg = r / PI;
        int lii = r % PI;
        Abuf[wp][lii][lg * KB + kb] = v;
      }
      // ---- emit results of tile n-2 ----
      if (n >= 2 && t < OUTBEATS) {
        int r = t / (PJ / LANES);
        int q = t % (PJ / LANES);
        int sg = r / PI;
        int sii = r % PI;
        vec_t o;
      out_lane:
        for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
          o[l] = Cbuf[sg][sii][q * LANES + l];
        }
        sC.write(o);
      }
      // ---- compute tile n-1 ----
      if (n >= 1) {
        int k = t / G;
        int g = t % G;
        int kb = k / LANES;
        int kl = k % LANES;
        float a[PI];
#pragma HLS ARRAY_PARTITION variable=a complete
        float b[PJ];
#pragma HLS ARRAY_PARTITION variable=b complete
      get_a:
        for (int ii = 0; ii < PI; ii++) {
#pragma HLS UNROLL
          vec_t av = Abuf[rp][ii][g * KB + kb];
          a[ii] = av[kl];
        }
      get_b:
        for (int tj = 0; tj < PJ; tj++) {
#pragma HLS UNROLL
          vec_t bv = Bres[tj][kb];
          b[tj] = bv[kl];
        }
      mac_i:
        for (int ii = 0; ii < PI; ii++) {
#pragma HLS UNROLL
        mac_j:
          for (int tj = 0; tj < PJ; tj++) {
#pragma HLS UNROLL
            float prod = a[ii] * b[tj];
            float old = (k == 0) ? 0.0f : acc[g][ii][tj];
            float s = old + prod;
            acc[g][ii][tj] = s;
            if (k == K - 1) Cbuf[g][ii][tj] = s;
          }
        }
      }
    }
  }
epilogue_out:
  for (int t = 0; t < OUTBEATS; t++) {
#pragma HLS PIPELINE II=1
    int r = t / (PJ / LANES);
    int q = t % (PJ / LANES);
    int sg = r / PI;
    int sii = r % PI;
    vec_t o;
  epi_lane:
    for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
      o[l] = Cbuf[sg][sii][q * LANES + l];
    }
    sC.write(o);
  }
}

static void store_C(hls::stream<vec_t> &sC, data_t C[I][J]) {
store_tile:
  for (int tl = 0; tl < NT; tl++) {
    const int i0 = (tl % TPP) * TROWS;
    const int j0 = (tl / TPP) * PJ;
  store_beat:
    for (int t = 0; t < OUTBEATS; t++) {
#pragma HLS PIPELINE II=1
      vec_t v = sC.read();
      int r = t / (PJ / LANES);
      int q = t % (PJ / LANES);
      if (j0 + q * LANES < J) {
      store_lane:
        for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
          C[i0 + r][j0 + q * LANES + l] = v[l];
        }
      }
    }
  }
}

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=1048576 max_widen_bitwidth=512 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS DATAFLOW
  hls::stream<vec_t> sA("sA");
  hls::stream<vec_t> sB("sB");
  hls::stream<vec_t> sC("sC");
#pragma HLS STREAM variable=sA depth=64
#pragma HLS STREAM variable=sB depth=64
#pragma HLS STREAM variable=sC depth=64
  load_A(A, sA);
  load_B(B, sB);
  compute(sA, sB, sC);
  store_C(sC, C);
}
