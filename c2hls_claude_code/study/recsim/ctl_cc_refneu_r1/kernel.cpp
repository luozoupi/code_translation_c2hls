#include "kernel.h"
#include <hls_stream.h>

// ---------------------------------------------------------------------------
//   PE array : TI x TJ = 32 x 32 = 1024 fp32 MACs / cycle
//   NB       : column blocks interleaved in the k pipeline (RAW distance NB
//              on each accumulator hides fadd latency)
//   JS       : resident B strip width = C tile width (512)
//   LANES    : floats per 512-bit beat
// ---------------------------------------------------------------------------
#define TI     32
#define TJ     32
#define NB     16
#define JS     (TJ * NB)
#define LANES  16
#define NT_I   (I / TI)
#define NT_J   (J / JS)
#define KB     (K / LANES)
#define CB     (JS / LANES)

struct beat_t {
  data_t d[LANES];
};

// ---------------------------- load A (gmem0) -------------------------------
static void load_A_task(data_t A[I][K], hls::stream<beat_t>& sA) {
load_A_strip:
  for (int tj = 0; tj < NT_J; tj++) {
  load_A_tile:
    for (int ti = 0; ti < NT_I; ti++) {
      int i0 = ti * TI;
    load_A_beat:
      for (int idx = 0; idx < TI * KB; idx++) {
#pragma HLS PIPELINE II=1
        int ii = idx / KB;
        int kb = (idx % KB) * LANES;
        beat_t b;
      load_A_lane:
        for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
          b.d[l] = A[i0 + ii][kb + l];
        }
        sA.write(b);
      }
    }
  }
}

// ---------------------------- load B (gmem1) -------------------------------
static void load_B_task(data_t B[J][K], hls::stream<beat_t>& sB) {
load_B_strip:
  for (int tj = 0; tj < NT_J; tj++) {
    int j0 = tj * JS;
  load_B_beat:
    for (int idx = 0; idx < JS * KB; idx++) {
#pragma HLS PIPELINE II=1
      int j  = idx / KB;
      int kb = (idx % KB) * LANES;
      beat_t b;
    load_B_lane:
      for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
        b.d[l] = B[j0 + j][kb + l];
      }
      sB.write(b);
    }
  }
}

// ---------------------------- load C (gmem2 read) --------------------------
static void load_C_task(data_t C[I][J], hls::stream<beat_t>& sC) {
load_C_strip:
  for (int tj = 0; tj < NT_J; tj++) {
    int j0 = tj * JS;
  load_C_tile:
    for (int ti = 0; ti < NT_I; ti++) {
      int i0 = ti * TI;
    load_C_row:
      for (int ii = 0; ii < TI; ii++) {
      load_C_beat:
        for (int c = 0; c < CB; c++) {
#pragma HLS PIPELINE II=1
          beat_t b;
        load_C_lane:
          for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
            b.d[l] = C[i0 + ii][j0 + c * LANES + l];
          }
          sC.write(b);
        }
      }
    }
  }
}

// ---------------------------- compute (on-chip only) -----------------------
static void compute_task(hls::stream<beat_t>& sA, hls::stream<beat_t>& sB,
                         hls::stream<beat_t>& sOut) {
  // Resident B strip, transposed: Bloc[k][jb][jj]
  data_t Bloc[K][NB][TJ];
#pragma HLS ARRAY_PARTITION variable=Bloc cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=Bloc complete dim=3

  // A tile, transposed: Aloc[k][ii]
  data_t Aloc[K][TI];
#pragma HLS ARRAY_PARTITION variable=Aloc cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=Aloc complete dim=2
#pragma HLS BIND_STORAGE variable=Aloc type=ram_2p impl=lutram

  // Accumulators: 1024 LUTRAMs of depth NB
  data_t acc[NB][TI][TJ];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=2
#pragma HLS ARRAY_PARTITION variable=acc complete dim=3
#pragma HLS BIND_STORAGE variable=acc type=ram_s2p impl=lutram

comp_strip:
  for (int tj = 0; tj < NT_J; tj++) {
  fill_B_beat:
    for (int idx = 0; idx < JS * KB; idx++) {
#pragma HLS PIPELINE II=1
      int j  = idx / KB;
      int kb = (idx % KB) * LANES;
      int jb = j / TJ;
      int jj = j % TJ;
      beat_t b = sB.read();
    fill_B_lane:
      for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
        Bloc[kb + l][jb][jj] = b.d[l];
      }
    }

  comp_tile:
    for (int ti = 0; ti < NT_I; ti++) {
    fill_A_beat:
      for (int idx = 0; idx < TI * KB; idx++) {
#pragma HLS PIPELINE II=1
        int ii = idx / KB;
        int kb = (idx % KB) * LANES;
        beat_t b = sA.read();
      fill_A_lane:
        for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
          Aloc[kb + l][ii] = b.d[l];
        }
      }

    compute_k:
      for (int k = 0; k < K; k++) {
      compute_jb:
        for (int jb = 0; jb < NB; jb++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=acc inter RAW distance=16 true
        pe_i:
          for (int ii = 0; ii < TI; ii++) {
#pragma HLS UNROLL
          pe_j:
            for (int jj = 0; jj < TJ; jj++) {
#pragma HLS UNROLL
              data_t prev = (k == 0) ? (data_t)0 : acc[jb][ii][jj];
              acc[jb][ii][jj] = prev + Aloc[k][ii] * Bloc[k][jb][jj];
            }
          }
        }
      }

    drain_beat:
      for (int idx = 0; idx < TI * CB; idx++) {
#pragma HLS PIPELINE II=1
        int ii = idx / CB;
        int c  = idx % CB;
        int jb = c / (TJ / LANES);
        int jo = (c % (TJ / LANES)) * LANES;
        beat_t b;
      drain_lane:
        for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
          b.d[l] = acc[jb][ii][jo + l];
        }
        sOut.write(b);
      }
    }
  }
}

// ---------------------------- store C (gmem2 write) ------------------------
static void store_C_task(data_t C[I][J], hls::stream<beat_t>& sOut,
                         hls::stream<beat_t>& sC) {
store_C_strip:
  for (int tj = 0; tj < NT_J; tj++) {
    int j0 = tj * JS;
  store_C_tile:
    for (int ti = 0; ti < NT_I; ti++) {
      int i0 = ti * TI;
    store_C_row:
      for (int ii = 0; ii < TI; ii++) {
      store_C_beat:
        for (int c = 0; c < CB; c++) {
#pragma HLS PIPELINE II=1
          beat_t r = sOut.read();
          beat_t o = sC.read();
        store_C_lane:
          for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
            C[i0 + ii][j0 + c * LANES + l] = o.d[l] + r.d[l];
          }
        }
      }
    }
  }
}

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=16 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS DATAFLOW
  hls::stream<beat_t> sA("sA");
  hls::stream<beat_t> sB("sB");
  hls::stream<beat_t> sC("sC");
  hls::stream<beat_t> sOut("sOut");
#pragma HLS STREAM variable=sA depth=64
#pragma HLS STREAM variable=sB depth=64
#pragma HLS STREAM variable=sC depth=128
#pragma HLS STREAM variable=sOut depth=128

  load_A_task(A, sA);
  load_B_task(B, sB);
  load_C_task(C, sC);
  compute_task(sA, sB, sOut);
  store_C_task(C, sOut, sC);
}