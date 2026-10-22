#include "kernel.h"

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
#define LN     16              // floats per 512-bit beat == k-lanes per PE
#define PE_I   8               // output rows computed in parallel
#define PE_J   16              // output cols computed in parallel
#define NKB    (K / LN)        // 64 k-blocks
#define NJB    (J / PE_J)      // 64 j-blocks
#define NT     (I / PE_I)      // 128 row tiles
#define TB     (PE_I * NKB)    // 512 beats per A tile (== beats per C tile)
#define NIT    (NKB * NJB)     // 4096 compute iterations per tile

#if (J != K) || (PE_J != LN)
#error "This kernel assumes J == K and PE_J == LN"
#endif

// B fully resident: B_loc[q][u][jb*NKB+kb] = B[jb*16+q][kb*16+u]
static data_t B_loc[PE_J][LN][NJB * NKB];
// A row tile: A_loc[p][u][kb] = A[t*8+p][kb*16+u]
static data_t A_loc[PE_I][LN][NKB];
// C accumulator tile: acc[p][q][jb] = C[t*8+p][jb*16+q]
static data_t acc[PE_I][PE_J][NJB];

// ---------------------------------------------------------------------------
// Load B once (512-bit beats)
// ---------------------------------------------------------------------------
static void load_B(data_t B[J][K]) {
load_B_row:
  for (int j = 0; j < J; j++) {
  load_B_beat:
    for (int kb = 0; kb < NKB; kb++) {
#pragma HLS PIPELINE II=1
    load_B_lane:
      for (int u = 0; u < LN; u++) {
#pragma HLS UNROLL
        B_loc[j % PE_J][u][(j / PE_J) * NKB + kb] = B[j][kb * LN + u];
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Load A tile + C tile (512-bit beats on separate bundles)
// ---------------------------------------------------------------------------
static void load_tile(data_t A[I][K], data_t C[I][J], int t) {
load_tile_beat:
  for (int x = 0; x < TB; x++) {
#pragma HLS PIPELINE II=1
    const int r = x / NKB;
    const int c = x % NKB;
  load_tile_lane:
    for (int u = 0; u < LN; u++) {
#pragma HLS UNROLL
      A_loc[r][u][c] = A[t * PE_I + r][c * LN + u];
      acc[r][u][c]   = C[t * PE_I + r][c * LN + u];
    }
  }
}

// ---------------------------------------------------------------------------
// Store C tile (512-bit beats)
// ---------------------------------------------------------------------------
static void store_tile(data_t C[I][J], int t) {
store_tile_beat:
  for (int x = 0; x < TB; x++) {
#pragma HLS PIPELINE II=1
    const int r = x / NJB;
    const int c = x % NJB;
  store_tile_lane:
    for (int u = 0; u < LN; u++) {
#pragma HLS UNROLL
      C[t * PE_I + r][c * LN + u] = acc[r][u][c];
    }
  }
}

// ---------------------------------------------------------------------------
// Compute one tile: 8x16 PEs, each a 16-lane dot product + accumulate
// ---------------------------------------------------------------------------
static void compute_tile() {
compute_it:
  for (int it = 0; it < NIT; it++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=acc inter false
    const int kb = it / NJB;
    const int jb = it % NJB;

    data_t a[PE_I][LN];
#pragma HLS ARRAY_PARTITION variable=a complete dim=0
    data_t b[PE_J][LN];
#pragma HLS ARRAY_PARTITION variable=b complete dim=0
  rd_a_p:
    for (int p = 0; p < PE_I; p++) {
#pragma HLS UNROLL
    rd_a_u:
      for (int u = 0; u < LN; u++) {
#pragma HLS UNROLL
        a[p][u] = A_loc[p][u][kb];
      }
    }
  rd_b_q:
    for (int q = 0; q < PE_J; q++) {
#pragma HLS UNROLL
    rd_b_u:
      for (int u = 0; u < LN; u++) {
#pragma HLS UNROLL
        b[q][u] = B_loc[q][u][jb * NKB + kb];
      }
    }

  pe_row:
    for (int p = 0; p < PE_I; p++) {
#pragma HLS UNROLL
    pe_col:
      for (int q = 0; q < PE_J; q++) {
#pragma HLS UNROLL
        data_t prod[LN];
#pragma HLS ARRAY_PARTITION variable=prod complete
      mac_mul:
        for (int u = 0; u < LN; u++) {
#pragma HLS UNROLL
          prod[u] = a[p][u] * b[q][u];
        }
        data_t s1[LN / 2];
#pragma HLS ARRAY_PARTITION variable=s1 complete
      add_l1:
        for (int v = 0; v < LN / 2; v++) {
#pragma HLS UNROLL
          data_t tt = prod[2 * v] + prod[2 * v + 1];
#pragma HLS BIND_OP variable=tt op=fadd impl=fabric
          s1[v] = tt;
        }
        data_t s2[LN / 4];
#pragma HLS ARRAY_PARTITION variable=s2 complete
      add_l2:
        for (int v = 0; v < LN / 4; v++) {
#pragma HLS UNROLL
          s2[v] = s1[2 * v] + s1[2 * v + 1];
        }
        data_t s3a = s2[0] + s2[1];
        data_t s3b = s2[2] + s2[3];
        data_t s4  = s3a + s3b;
        acc[p][q][jb] = acc[p][q][jb] + s4;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Top
// ---------------------------------------------------------------------------
extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 max_widen_bitwidth=512 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=16 num_write_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 max_widen_bitwidth=512 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=16 num_write_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 max_widen_bitwidth=512 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=16 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS ARRAY_PARTITION variable=B_loc complete dim=1
#pragma HLS ARRAY_PARTITION variable=B_loc complete dim=2
#pragma HLS BIND_STORAGE variable=B_loc type=ram_2p impl=uram
#pragma HLS ARRAY_PARTITION variable=A_loc complete dim=1
#pragma HLS ARRAY_PARTITION variable=A_loc complete dim=2
#pragma HLS BIND_STORAGE variable=A_loc type=ram_2p impl=lutram
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc complete dim=2
#pragma HLS BIND_STORAGE variable=acc type=ram_2p impl=lutram

  load_B(B);

tile_loop:
  for (int t = 0; t < NT; t++) {
    load_tile(A, C, t);
    compute_tile();
    store_tile(C, t);
  }
}