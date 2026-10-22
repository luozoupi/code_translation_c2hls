#include "kernel.h"

// Output-stationary PE array: PI rows x PJ cols of fp32 MACs per cycle.
#define PI 52                    // parallel output rows (i)
#define PJ 32                    // parallel output cols (j)
#define NB (J / PJ)              // j-blocks per row (32) = acc recurrence distance
#define LANES 16                 // floats per 512-bit beat
#define KW (K / LANES)           // beats per row (64)
#define NIB ((I + PI - 1) / PI)  // i-blocks (20)
#define LAST_ROWS (I - (NIB - 1) * PI)  // rows in final i-block (4)

// Fixed-trip row-block loader: A rows -> a_buf, C rows -> acc (512-bit bursts).
template <int ROWS>
static void load_AC_rows(const data_t *Af, const data_t *Cf, data_t a_buf[PI][KW][LANES],
                         data_t acc[PI][PJ][NB], int i0) {
#pragma HLS INLINE
load_AC:
  for (int t = 0; t < ROWS * KW; t++) {
#pragma HLS PIPELINE II=1
    int pi = t / KW;
    int kk = t % KW;
    int base = (i0 * KW + t) * LANES;
  load_AC_lane:
    for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
      a_buf[pi][kk][l] = Af[base + l];
      acc[pi][(kk & 1) * LANES + l][kk >> 1] = (data_t)0;
    }
  }
}

// Fixed-trip row-block store: acc -> C rows (512-bit bursts).
template <int ROWS>
static void store_C_rows(data_t *Cf, data_t acc[PI][PJ][NB], int i0) {
#pragma HLS INLINE
store_C:
  for (int t = 0; t < ROWS * KW; t++) {
#pragma HLS PIPELINE II=1
    int pi = t / KW;
    int kk = t % KW;
    int base = (i0 * KW + t) * LANES;
  store_C_lane:
    for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
      Cf[base + l] = acc[pi][(kk & 1) * LANES + l][kk >> 1];
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

  // Full B on-chip: bank pj holds rows j with j%PJ==pj; 16 k-lanes packed per word.
  data_t b_buf[PJ][NB * KW][LANES];
#pragma HLS ARRAY_PARTITION variable=b_buf complete dim=1
#pragma HLS ARRAY_RESHAPE variable=b_buf complete dim=3
#pragma HLS BIND_STORAGE variable=b_buf type=ram_2p impl=uram

  // A row block: PI rows, 16 k-lanes packed per word.
  data_t a_buf[PI][KW][LANES];
#pragma HLS ARRAY_PARTITION variable=a_buf complete dim=1
#pragma HLS ARRAY_RESHAPE variable=a_buf complete dim=3
#pragma HLS BIND_STORAGE variable=a_buf type=ram_2p impl=lutram

  // Accumulators: acc[pi][pj][jb] holds C[i0+pi][jb*PJ+pj].
  data_t acc[PI][PJ][NB];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc complete dim=2
#pragma HLS BIND_STORAGE variable=acc type=ram_2p impl=lutram

  const data_t *Af = &A[0][0];
  const data_t *Bf = &B[0][0];
  data_t *Cf = &C[0][0];

load_B:
  for (int t = 0; t < J * KW; t++) {
#pragma HLS PIPELINE II=1
    int j = t / KW;
    int kk = t % KW;
  load_B_lane:
    for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
      b_buf[j % PJ][(j / PJ) * KW + kk][l] = Bf[t * LANES + l];
    }
  }

i_blk:
  for (int ib = 0; ib < NIB; ib++) {
    const int i0 = ib * PI;
    const bool full = (ib < NIB - 1);

    if (full) load_AC_rows<PI>(Af, Cf, a_buf, acc, i0);
    else      load_AC_rows<LAST_ROWS>(Af, Cf, a_buf, acc, i0);

    // C row-block += A row-block * B^T ; k-order accumulation matches reference.
  compute_k:
    for (int k = 0; k < K; k++) {
    compute_jb:
      for (int jb = 0; jb < NB; jb++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=acc inter false
        const int kw = k / LANES;
        const int ln = k % LANES;
        data_t av[PI];
#pragma HLS ARRAY_PARTITION variable=av complete
        data_t bv[PJ];
#pragma HLS ARRAY_PARTITION variable=bv complete
      rd_a:
        for (int pi = 0; pi < PI; pi++) {
#pragma HLS UNROLL
          av[pi] = a_buf[pi][kw][ln];
        }
      rd_b:
        for (int pj = 0; pj < PJ; pj++) {
#pragma HLS UNROLL
          bv[pj] = b_buf[pj][jb * KW + kw][ln];
        }
      mac_pi:
        for (int pi = 0; pi < PI; pi++) {
#pragma HLS UNROLL
        mac_pj:
          for (int pj = 0; pj < PJ; pj++) {
#pragma HLS UNROLL
            acc[pi][pj][jb] = acc[pi][pj][jb] + av[pi] * bv[pj];
          }
        }
      }
    }

    if (full) store_C_rows<PI>(Cf, acc, i0);
    else      store_C_rows<LAST_ROWS>(Cf, acc, i0);
  }
}