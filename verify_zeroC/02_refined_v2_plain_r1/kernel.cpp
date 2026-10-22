#include "kernel.h"

#if defined(__has_include)
#if __has_include(<hls_stream.h>)
#include <hls_stream.h>
#define AUTOSA_HAVE_HLS_STREAM 1
#endif
#endif

#ifndef AUTOSA_HAVE_HLS_STREAM
#include <deque>
namespace hls {
template <typename T>
class stream {
 public:
  stream() {}
  stream(const char *) {}
  void write(const T &v) { q_.push_back(v); }
  T read() {
    T v = q_.front();
    q_.pop_front();
    return v;
  }
  bool empty() const { return q_.empty(); }
 private:
  std::deque<T> q_;
};
}  // namespace hls
#endif

// ---------------------------------------------------------------------------
// Datapath geometry (U280 sized): P_I x P_J x KU = 8 x 13 x 16 = 1664 fp32 MACs
// (~80 DSP per PE -> ~8320 DSP, ~92% of 9024)
// ---------------------------------------------------------------------------
#define P_I      8                      // PE rows (A rows per cycle)
#define P_J      13                     // PE cols (B rows per cycle)
#define KU       16                     // k lanes per cycle (= one 512-bit beat)
#define LANES    16                     // floats per 512-bit beat
#define TI       128                    // A rows resident on chip
#define NJB      16                     // j-blocks per tile
#define TJ       (P_J * NJB)            // 208 B rows per tile (multiple of 16)
#define NIB      (TI / P_I)             // 16
#define KB       (K / KU)               // 64 k-blocks
#define NPOS     (NIB * NJB)            // 256 output positions (acc reuse distance)
#define N_ITILE  (I / TI)               // 8
#define N_JTILE  ((J + TJ - 1) / TJ)    // 5 (last tile partial: 192 cols)
#define NT       (N_ITILE * N_JTILE)    // 40 tiles
#define SLOT     (KB * NPOS)            // 16384 compute iterations per tile
#define A_HALF   (NIB * KB)             // 1024 words per A half per bank
#define B_HALF   (NJB * KB)             // 1024 words per B half per bank
#define A_BEATS  (TI * KB)              // 8192 beats per A block
#define B_BEATS  (TJ * KB)              // 13312 beats per B tile
#define C_PACKS  (TI * NJB)             // 2048 packs of 13 floats per C tile
#define C_ROWB   (TJ / LANES)           // 13 beats per C tile row
#define C_LOAD_START 8192               // compute-slot offset of C-tile intake

struct pk16 { data_t v[LANES]; };
struct pkj  { data_t v[P_J]; };

// ---------------------------------------------------------------------------
// A reader: one 128-row block per i-tile, 16 floats per beat
// ---------------------------------------------------------------------------
static void read_A(data_t A[I][K], hls::stream<pk16> &sA) {
  rdA_block: for (int ib = 0; ib < N_ITILE; ++ib) {
    rdA_beat: for (int idx = 0; idx < A_BEATS; ++idx) {
#pragma HLS PIPELINE II=1
      int r  = idx / KB;
      int kb = idx % KB;
      pk16 p;
      rdA_lane: for (int l = 0; l < LANES; ++l) {
#pragma HLS UNROLL
        p.v[l] = A[ib * TI + r][kb * KU + l];
      }
      sA.write(p);
    }
  }
}

// ---------------------------------------------------------------------------
// B reader: one 208-row tile per (i,j) tile; rows past J are clamped (unused)
// ---------------------------------------------------------------------------
static void read_B(data_t B[J][K], hls::stream<pk16> &sB) {
  rdB_tile: for (int t = 0; t < NT; ++t) {
    int jt = t % N_JTILE;
    rdB_beat: for (int idx = 0; idx < B_BEATS; ++idx) {
#pragma HLS PIPELINE II=1
      int c  = idx / KB;
      int kb = idx % KB;
      int j  = jt * TJ + c;
      int jr = (j < J) ? j : (J - 1);
      pk16 p;
      rdB_lane: for (int l = 0; l < LANES; ++l) {
#pragma HLS UNROLL
        p.v[l] = B[jr][kb * KU + l];
      }
      sB.write(p);
    }
  }
}

// ---------------------------------------------------------------------------
// C reader: 512-bit row reads into a tile buffer, re-packed into 13-wide packs
// ---------------------------------------------------------------------------
static void read_C(data_t C[I][J], hls::stream<pkj> &sC) {
  static data_t ctile[TI][TJ];
#pragma HLS ARRAY_PARTITION variable=ctile complete dim=2
  rdC_tile: for (int u = 0; u < NT; ++u) {
    int ib = u / N_JTILE;
    int jt = u % N_JTILE;
    int rem = J - jt * TJ;
    int nb = (rem >= TJ) ? C_ROWB : (rem / LANES);
    int nbeats = TI * nb;
    int r = 0, b = 0;
    rdC_beat: for (int idx = 0; idx < nbeats; ++idx) {
#pragma HLS LOOP_TRIPCOUNT min=1536 max=1664
#pragma HLS PIPELINE II=1
      rdC_lane: for (int l = 0; l < LANES; ++l) {
#pragma HLS UNROLL
        ctile[r][b * LANES + l] = (data_t)0;
      }
      if (b == nb - 1) { b = 0; ++r; } else { ++b; }
    }
    rdC_push: for (int p = 0; p < C_PACKS; ++p) {
#pragma HLS PIPELINE II=1
      int rr  = p / NJB;
      int pjb = p % NJB;
      pkj pk;
      rdC_pack: for (int pj = 0; pj < P_J; ++pj) {
#pragma HLS UNROLL
        pk.v[pj] = ctile[rr][pjb * P_J + pj];
      }
      sC.write(pk);
    }
  }
}

// ---------------------------------------------------------------------------
// C writer: collect 13-wide packs, write valid columns as 512-bit beats
// ---------------------------------------------------------------------------
static void write_C(data_t C[I][J], hls::stream<pkj> &sO) {
  static data_t otile[TI][TJ];
#pragma HLS ARRAY_PARTITION variable=otile complete dim=2
  wrC_tile: for (int u = 0; u < NT; ++u) {
    int ib = u / N_JTILE;
    int jt = u % N_JTILE;
    wrC_pop: for (int p = 0; p < C_PACKS; ++p) {
#pragma HLS PIPELINE II=1
      int rr  = p / NJB;
      int pjb = p % NJB;
      pkj pk = sO.read();
      wrC_pack: for (int pj = 0; pj < P_J; ++pj) {
#pragma HLS UNROLL
        otile[rr][pjb * P_J + pj] = pk.v[pj];
      }
    }
    int rem = J - jt * TJ;
    int nb = (rem >= TJ) ? C_ROWB : (rem / LANES);
    int nbeats = TI * nb;
    int r = 0, b = 0;
    wrC_beat: for (int idx = 0; idx < nbeats; ++idx) {
#pragma HLS LOOP_TRIPCOUNT min=1536 max=1664
#pragma HLS PIPELINE II=1
      wrC_lane: for (int l = 0; l < LANES; ++l) {
#pragma HLS UNROLL
        C[ib * TI + r][jt * TJ + b * LANES + l] = otile[r][b * LANES + l];
      }
      if (b == nb - 1) { b = 0; ++r; } else { ++b; }
    }
  }
}

// ---------------------------------------------------------------------------
// 16-lane dot product as a balanced adder tree
// ---------------------------------------------------------------------------
static data_t dot16(const data_t a[KU], const data_t b[KU]) {
#pragma HLS INLINE
  data_t p[KU];
#pragma HLS ARRAY_PARTITION variable=p complete
  dot_mul: for (int q = 0; q < KU; ++q) {
#pragma HLS UNROLL
    p[q] = a[q] * b[q];
  }
  data_t s8[8];
#pragma HLS ARRAY_PARTITION variable=s8 complete
  dot_l1: for (int q = 0; q < 8; ++q) {
#pragma HLS UNROLL
    s8[q] = p[2 * q] + p[2 * q + 1];
  }
  data_t s4[4];
#pragma HLS ARRAY_PARTITION variable=s4 complete
  dot_l2: for (int q = 0; q < 4; ++q) {
#pragma HLS UNROLL
    s4[q] = s8[2 * q] + s8[2 * q + 1];
  }
  data_t s2a = s4[0] + s4[1];
  data_t s2b = s4[2] + s4[3];
  return s2a + s2b;
}

// ---------------------------------------------------------------------------
// Compute engine: one II=1 pipeline per tile slot; loads tile t, computes
// tile t-1 and drains tile t-2 concurrently (ping-pong halves in each buffer).
// ---------------------------------------------------------------------------
static void compute(hls::stream<pk16> &sA, hls::stream<pk16> &sB,
                    hls::stream<pkj> &sC, hls::stream<pkj> &sO) {
  static data_t A_loc[P_I][KU][2 * A_HALF];
#pragma HLS ARRAY_PARTITION variable=A_loc complete dim=1
#pragma HLS ARRAY_PARTITION variable=A_loc complete dim=2
  static data_t B_loc[P_J][KU][2 * B_HALF];
#pragma HLS ARRAY_PARTITION variable=B_loc complete dim=1
#pragma HLS ARRAY_PARTITION variable=B_loc complete dim=2
  static data_t Cbuf[P_I][P_J][2 * NPOS];
#pragma HLS ARRAY_PARTITION variable=Cbuf complete dim=1
#pragma HLS ARRAY_PARTITION variable=Cbuf complete dim=2
  static data_t Obuf[P_I][P_J][2 * NPOS];
#pragma HLS ARRAY_PARTITION variable=Obuf complete dim=1
#pragma HLS ARRAY_PARTITION variable=Obuf complete dim=2
  static data_t acc[P_I][P_J][NPOS];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc complete dim=2

  slot_loop: for (int t = 0; t < NT + 2; ++t) {
    const bool do_load  = (t < NT);
    const bool do_comp  = (t >= 1) && (t <= NT);
    const bool do_drain = (t >= 2);
    const int  ib_ld    = t / N_JTILE;
    const bool load_a   = do_load && ((t % N_JTILE) == 0);
    const int  ib_cp    = (t >= 1) ? ((t - 1) / N_JTILE) : 0;

    const int a_ld_off = (ib_ld & 1) * A_HALF;
    const int a_cp_off = (ib_cp & 1) * A_HALF;
    const int b_ld_off = (t & 1) * B_HALF;
    const int b_cp_off = ((t + 1) & 1) * B_HALF;
    const int c_ld_off = (t & 1) * NPOS;
    const int c_cp_off = ((t + 1) & 1) * NPOS;
    const int o_cp_off = ((t + 1) & 1) * NPOS;
    const int o_dr_off = (t & 1) * NPOS;

    slot_iter: for (int it = 0; it < SLOT; ++it) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=A_loc type=inter dependent=false
#pragma HLS DEPENDENCE variable=A_loc type=intra dependent=false
#pragma HLS DEPENDENCE variable=B_loc type=inter dependent=false
#pragma HLS DEPENDENCE variable=B_loc type=intra dependent=false
#pragma HLS DEPENDENCE variable=Cbuf type=inter dependent=false
#pragma HLS DEPENDENCE variable=Cbuf type=intra dependent=false
#pragma HLS DEPENDENCE variable=Obuf type=inter dependent=false
#pragma HLS DEPENDENCE variable=Obuf type=intra dependent=false
#pragma HLS DEPENDENCE variable=acc type=inter direction=RAW dependent=true distance=256

      // ---------------- load A block (next i-block, other half) ----------------
      if (load_a && it < A_BEATS) {
        pk16 pa = sA.read();
        int r   = it / KB;
        int kbl = it % KB;
        int pi  = r % P_I;
        int pib = r / P_I;
        ldA_lane: for (int ku = 0; ku < KU; ++ku) {
#pragma HLS UNROLL
          A_loc[pi][ku][a_ld_off + pib * KB + kbl] = pa.v[ku];
        }
      }
      // ---------------- load B tile t (other half) ----------------
      if (do_load && it < B_BEATS) {
        pk16 pb = sB.read();
        int c   = it / KB;
        int kbl = it % KB;
        int pjb = c / P_J;
        int pj  = c % P_J;
        ldB_lane: for (int ku = 0; ku < KU; ++ku) {
#pragma HLS UNROLL
          B_loc[pj][ku][b_ld_off + pjb * KB + kbl] = pb.v[ku];
        }
      }
      // ---------------- load C tile t (initial accumulator values) ----------------
      if (do_load && it >= C_LOAD_START && it < C_LOAD_START + C_PACKS) {
        pkj pc = sC.read();
        int p   = it - C_LOAD_START;
        int r   = p / NJB;
        int pjb = p % NJB;
        int pi  = r % P_I;
        int pib = r / P_I;
        int ps  = pib * NJB + pjb;
        ldC_lane: for (int pj = 0; pj < P_J; ++pj) {
#pragma HLS UNROLL
          Cbuf[pi][pj][c_ld_off + ps] = pc.v[pj];
        }
      }
      // ---------------- drain results of tile t-2 ----------------
      if (do_drain && it < C_PACKS) {
        int p   = it;
        int r   = p / NJB;
        int pjb = p % NJB;
        int pi  = r % P_I;
        int pib = r / P_I;
        int ps  = pib * NJB + pjb;
        pkj po;
        drain_lane: for (int pj = 0; pj < P_J; ++pj) {
#pragma HLS UNROLL
          po.v[pj] = Obuf[pi][pj][o_dr_off + ps];
        }
        sO.write(po);
      }
      // ---------------- compute tile t-1: 8 x 13 x 16 MACs ----------------
      int kb  = it / NPOS;
      int pos = it % NPOS;
      int pib = pos / NJB;
      int pjb = pos % NJB;

      data_t a_reg[P_I][KU];
#pragma HLS ARRAY_PARTITION variable=a_reg complete dim=0
      data_t b_reg[P_J][KU];
#pragma HLS ARRAY_PARTITION variable=b_reg complete dim=0
      rd_a_pi: for (int pi = 0; pi < P_I; ++pi) {
#pragma HLS UNROLL
        rd_a_ku: for (int ku = 0; ku < KU; ++ku) {
#pragma HLS UNROLL
          a_reg[pi][ku] = A_loc[pi][ku][a_cp_off + pib * KB + kb];
        }
      }
      rd_b_pj: for (int pj = 0; pj < P_J; ++pj) {
#pragma HLS UNROLL
        rd_b_ku: for (int ku = 0; ku < KU; ++ku) {
#pragma HLS UNROLL
          b_reg[pj][ku] = B_loc[pj][ku][b_cp_off + pjb * KB + kb];
        }
      }
      pe_row: for (int pi = 0; pi < P_I; ++pi) {
#pragma HLS UNROLL
        pe_col: for (int pj = 0; pj < P_J; ++pj) {
#pragma HLS UNROLL
          data_t s    = dot16(a_reg[pi], b_reg[pj]);
          data_t cin  = Cbuf[pi][pj][c_cp_off + pos];
          data_t ain  = acc[pi][pj][pos];
          data_t base = (kb == 0) ? cin : ain;
          data_t v    = base + s;
          if (do_comp) {
            if (kb == KB - 1) {
              Obuf[pi][pj][o_cp_off + pos] = v;
            } else {
              acc[pi][pj][pos] = v;
            }
          }
        }
      }
    }
  }
}

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS DATAFLOW
  hls::stream<pk16> sA("sA");
  hls::stream<pk16> sB("sB");
  hls::stream<pkj>  sC("sC");
  hls::stream<pkj>  sO("sO");
#pragma HLS STREAM variable=sA depth=64
#pragma HLS STREAM variable=sB depth=64
#pragma HLS STREAM variable=sC depth=64
#pragma HLS STREAM variable=sO depth=64

  read_A(A, sA);
  read_B(B, sB);
  read_C(C, sC);
  compute(sA, sB, sC, sO);
  write_C(C, sO);
}