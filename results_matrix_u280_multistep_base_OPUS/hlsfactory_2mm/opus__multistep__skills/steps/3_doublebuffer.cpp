#include "2mm.h"
#include <cstring>

#define TILE_I 8

// Load this tile's inputs into the selected buffer set.
static void load_tile(int it, int tile_rows, int nk, int nl,
                      double A[NI][NK], double D[NI][NL],
                      double l_A_1[TILE_I][NK], double l_A_2[TILE_I][NK],
                      double l_D_1[TILE_I][NL], double l_D_2[TILE_I][NL],
                      int flag)
{
  int ii, k, j;
  if (flag == 0) {
    LOAD_A_I_0: for (ii = 0; ii < tile_rows; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=8
      LOAD_A_K_0: for (k = 0; k < nk; k++) {
#pragma HLS PIPELINE II=1
        l_A_1[ii][k] = A[it + ii][k];
      }
    }
    LOAD_D_I_0: for (ii = 0; ii < tile_rows; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=8
      LOAD_D_L_0: for (j = 0; j < nl; j++) {
#pragma HLS PIPELINE II=1
        l_D_1[ii][j] = D[it + ii][j];
      }
    }
  } else {
    LOAD_A_I_1: for (ii = 0; ii < tile_rows; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=8
      LOAD_A_K_1: for (k = 0; k < nk; k++) {
#pragma HLS PIPELINE II=1
        l_A_2[ii][k] = A[it + ii][k];
      }
    }
    LOAD_D_I_1: for (ii = 0; ii < tile_rows; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=8
      LOAD_D_L_1: for (j = 0; j < nl; j++) {
#pragma HLS PIPELINE II=1
        l_D_2[ii][j] = D[it + ii][j];
      }
    }
  }
}

// Compute on the selected buffer set.
static void compute_tile(double alpha, double beta,
                         int nj, int nk, int nl,
                         double l_A_1[TILE_I][NK], double l_A_2[TILE_I][NK],
                         double l_B[NK][NJ], double l_C[NJ][NL],
                         double l_tmp_1[TILE_I][NJ], double l_tmp_2[TILE_I][NJ],
                         double l_D_1[TILE_I][NL], double l_D_2[TILE_I][NL],
                         int flag)
{
  int ii, j, k;
  if (flag == 0) {
    MM1_J_0: for (j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
      MM1_I_0: for (ii = 0; ii < TILE_I; ii++) {
#pragma HLS UNROLL
        double acc = 0.0;
        MM1_K_0: for (k = 0; k < nk; ++k) {
#pragma HLS UNROLL factor=2
          acc += alpha * l_A_1[ii][k] * l_B[k][j];
        }
        l_tmp_1[ii][j] = acc;
      }
    }
    MM2_J_0: for (j = 0; j < nl; j++) {
#pragma HLS PIPELINE II=1
      MM2_I_0: for (ii = 0; ii < TILE_I; ii++) {
#pragma HLS UNROLL
        double acc = l_D_1[ii][j] * beta;
        MM2_K_0: for (k = 0; k < nj; ++k) {
#pragma HLS UNROLL factor=2
          acc += l_tmp_1[ii][k] * l_C[k][j];
        }
        l_D_1[ii][j] = acc;
      }
    }
  } else {
    MM1_J_1: for (j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
      MM1_I_1: for (ii = 0; ii < TILE_I; ii++) {
#pragma HLS UNROLL
        double acc = 0.0;
        MM1_K_1: for (k = 0; k < nk; ++k) {
#pragma HLS UNROLL factor=2
          acc += alpha * l_A_2[ii][k] * l_B[k][j];
        }
        l_tmp_2[ii][j] = acc;
      }
    }
    MM2_J_1: for (j = 0; j < nl; j++) {
#pragma HLS PIPELINE II=1
      MM2_I_1: for (ii = 0; ii < TILE_I; ii++) {
#pragma HLS UNROLL
        double acc = l_D_2[ii][j] * beta;
        MM2_K_1: for (k = 0; k < nj; ++k) {
#pragma HLS UNROLL factor=2
          acc += l_tmp_2[ii][k] * l_C[k][j];
        }
        l_D_2[ii][j] = acc;
      }
    }
  }
}

// Store the selected buffer set's results back to global memory.
static void store_tile(int it, int tile_rows, int nj, int nl,
                       double tmp[NI][NJ], double D[NI][NL],
                       double l_tmp_1[TILE_I][NJ], double l_tmp_2[TILE_I][NJ],
                       double l_D_1[TILE_I][NL], double l_D_2[TILE_I][NL],
                       int flag)
{
  int ii, j;
  if (flag == 0) {
    STORE_TMP_I_0: for (ii = 0; ii < tile_rows; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=8
      STORE_TMP_J_0: for (j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
        tmp[it + ii][j] = l_tmp_1[ii][j];
      }
    }
    STORE_D_I_0: for (ii = 0; ii < tile_rows; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=8
      STORE_D_J_0: for (j = 0; j < nl; j++) {
#pragma HLS PIPELINE II=1
        D[it + ii][j] = l_D_1[ii][j];
      }
    }
  } else {
    STORE_TMP_I_1: for (ii = 0; ii < tile_rows; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=8
      STORE_TMP_J_1: for (j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
        tmp[it + ii][j] = l_tmp_2[ii][j];
      }
    }
    STORE_D_I_1: for (ii = 0; ii < tile_rows; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=8
      STORE_D_J_1: for (j = 0; j < nl; j++) {
#pragma HLS PIPELINE II=1
        D[it + ii][j] = l_D_2[ii][j];
      }
    }
  }
}

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

  int j, k;

  // Stationary working sets that are reused across all i-tiles.
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

  // ---- Double-buffered per-tile local buffers (ping-pong sets) ----
  double l_A_1[TILE_I][NK];
#pragma HLS ARRAY_PARTITION variable=l_A_1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=l_A_1 cyclic factor=8 dim=2
  double l_A_2[TILE_I][NK];
#pragma HLS ARRAY_PARTITION variable=l_A_2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=l_A_2 cyclic factor=8 dim=2

  double l_tmp_1[TILE_I][NJ];
#pragma HLS ARRAY_PARTITION variable=l_tmp_1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=l_tmp_1 cyclic factor=8 dim=2
  double l_tmp_2[TILE_I][NJ];
#pragma HLS ARRAY_PARTITION variable=l_tmp_2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=l_tmp_2 cyclic factor=8 dim=2

  double l_D_1[TILE_I][NL];
#pragma HLS ARRAY_PARTITION variable=l_D_1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=l_D_1 cyclic factor=8 dim=2
  double l_D_2[TILE_I][NL];
#pragma HLS ARRAY_PARTITION variable=l_D_2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=l_D_2 cyclic factor=8 dim=2

  // Number of tiles along the i dimension.
  const int num_tiles = (ni + TILE_I - 1) / TILE_I;

  // Software-pipelined loop over tiles: overlap load(k+1) with compute/store(k).
  // We iterate one extra step to drain the last tile's compute/store.
  TILE_LOOP: for (int t = 0; t < num_tiles + 1; t++) {

    // ---- LOAD phase: stage tile t (if it exists) ----
    if (t < num_tiles) {
      int it_load = t * TILE_I;
      int rows_load = (it_load + TILE_I <= ni) ? TILE_I : (ni - it_load);
      int load_flag = t & 1;
      load_tile(it_load, rows_load, nk, nl,
                A, D,
                l_A_1, l_A_2, l_D_1, l_D_2,
                load_flag);
    }

    // ---- COMPUTE + STORE phase: process tile t-1 (if it exists) ----
    if (t > 0) {
      int it_comp = (t - 1) * TILE_I;
      int rows_comp = (it_comp + TILE_I <= ni) ? TILE_I : (ni - it_comp);
      int comp_flag = (t - 1) & 1;
      compute_tile(alpha, beta, nj, nk, nl,
                   l_A_1, l_A_2, l_B, l_C,
                   l_tmp_1, l_tmp_2, l_D_1, l_D_2,
                   comp_flag);
      store_tile(it_comp, rows_comp, nj, nl,
                 tmp, D,
                 l_tmp_1, l_tmp_2, l_D_1, l_D_2,
                 comp_flag);
    }
  }
}