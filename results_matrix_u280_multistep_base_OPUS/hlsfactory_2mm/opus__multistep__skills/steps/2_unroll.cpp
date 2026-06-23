#include "2mm.h"
#include <cstring>

#define TILE_I 8

void kernel_2mm(   
		double alpha,
		double beta,
		double tmp[ NI + 0][NJ + 0],
		double A[ NI + 0][NK + 0],
		double B[ NK + 0][NJ + 0],
		double C[ NJ + 0][NL + 0],
		double D[ NI + 0][NL + 0])
{
#pragma HLS INTERFACE m_axi port=tmp offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=A   offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=B   offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=C   offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=D   offset=slave bundle=gmem4
#pragma HLS INTERFACE s_axilite port=alpha bundle=control
#pragma HLS INTERFACE s_axilite port=beta  bundle=control
#pragma HLS INTERFACE s_axilite port=tmp   bundle=control
#pragma HLS INTERFACE s_axilite port=A     bundle=control
#pragma HLS INTERFACE s_axilite port=B     bundle=control
#pragma HLS INTERFACE s_axilite port=C     bundle=control
#pragma HLS INTERFACE s_axilite port=D     bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  const int ni = NI;
  const int nj = NJ;
  const int nk = NK;
  const int nl = NL;

  int i, j, k, ii;

  // Stationary working sets that are reused across all i-tiles.
  // B (NK x NJ) and C (NJ x NL) are fully reused by every row tile,
  // so we stage them once into local memory.
  double l_B[NK][NJ];
#pragma HLS ARRAY_PARTITION variable=l_B cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_B cyclic factor=2 dim=2
  double l_C[NJ][NL];
#pragma HLS ARRAY_PARTITION variable=l_C cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_C cyclic factor=2 dim=2

  // Load stationary inputs once.
  LOAD_B_K: for (k = 0; k < nk; k++)
    LOAD_B_J: for (j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
      l_B[k][j] = B[k][j];
    }

  LOAD_C_K: for (k = 0; k < nj; k++)
    LOAD_C_L: for (j = 0; j < nl; j++) {
#pragma HLS PIPELINE II=1
      l_C[k][j] = C[k][j];
    }

  // Process the row dimension (i) in tiles of TILE_I rows.
  TILE_LOOP: for (int it = 0; it < ni; it += TILE_I) {

    int tile_rows = (it + TILE_I <= ni) ? TILE_I : (ni - it);

    // Per-tile local buffers (small working set staged from global memory).
    double l_A[TILE_I][NK];
#pragma HLS ARRAY_PARTITION variable=l_A complete dim=1
#pragma HLS ARRAY_PARTITION variable=l_A cyclic factor=8 dim=2
    double l_tmp[TILE_I][NJ];
#pragma HLS ARRAY_PARTITION variable=l_tmp complete dim=1
#pragma HLS ARRAY_PARTITION variable=l_tmp cyclic factor=8 dim=2
    double l_D[TILE_I][NL];
#pragma HLS ARRAY_PARTITION variable=l_D complete dim=1
#pragma HLS ARRAY_PARTITION variable=l_D cyclic factor=8 dim=2

    // ---- LOAD phase: stage this tile's inputs into local buffers ----
    LOAD_A_I: for (ii = 0; ii < tile_rows; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=8
      LOAD_A_K: for (k = 0; k < nk; k++) {
#pragma HLS PIPELINE II=1
        l_A[ii][k] = A[it + ii][k];
      }
    }

    LOAD_D_I: for (ii = 0; ii < tile_rows; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=8
      LOAD_D_L: for (j = 0; j < nl; j++) {
#pragma HLS PIPELINE II=1
        l_D[ii][j] = D[it + ii][j];
      }
    }

    // ---- COMPUTE phase: operate entirely on local buffers ----

    // First matrix multiply: tmp = alpha * A * B (for this tile of rows)
    MM1_J: for (j = 0; j < nj; j++)
      {
#pragma HLS PIPELINE II=1
        MM1_I: for (ii = 0; ii < TILE_I; ii++) {
#pragma HLS UNROLL
          double acc = 0.0;
          MM1_K: for (k = 0; k < nk; ++k) {
#pragma HLS UNROLL factor=2
            acc += alpha * l_A[ii][k] * l_B[k][j];
          }
          l_tmp[ii][j] = acc;
        }
      }

    // Second matrix multiply: D = beta * D + tmp * C (for this tile of rows)
    MM2_J: for (j = 0; j < nl; j++)
      {
#pragma HLS PIPELINE II=1
        MM2_I: for (ii = 0; ii < TILE_I; ii++) {
#pragma HLS UNROLL
          double acc = l_D[ii][j] * beta;
          MM2_K: for (k = 0; k < nj; ++k) {
#pragma HLS UNROLL factor=2
            acc += l_tmp[ii][k] * l_C[k][j];
          }
          l_D[ii][j] = acc;
        }
      }

    // ---- STORE phase: write this tile's results back to global memory ----
    STORE_TMP_I: for (ii = 0; ii < tile_rows; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=8
      STORE_TMP_J: for (j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
        tmp[it + ii][j] = l_tmp[ii][j];
      }
    }

    STORE_D_I: for (ii = 0; ii < tile_rows; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=8
      STORE_D_J: for (j = 0; j < nl; j++) {
#pragma HLS PIPELINE II=1
        D[it + ii][j] = l_D[ii][j];
      }
    }
  }
}