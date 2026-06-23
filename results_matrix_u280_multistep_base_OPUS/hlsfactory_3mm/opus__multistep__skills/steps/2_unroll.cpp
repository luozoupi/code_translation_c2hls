#include "3mm.h"
#include <string.h>

#define TILE 256

void kernel_3mm(    
		double E[ NI + 0][NJ + 0],
		double A[ NI + 0][NK + 0],
		double B[ NK + 0][NJ + 0],
		double F[ NJ + 0][NL + 0],
		double C[ NJ + 0][NM + 0],
		double D[ NM + 0][NL + 0],
		double G[ NI + 0][NL + 0])
{
#pragma HLS INTERFACE m_axi port=E offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=F offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=D offset=slave bundle=gmem5
#pragma HLS INTERFACE m_axi port=G offset=slave bundle=gmem6
#pragma HLS INTERFACE s_axilite port=E bundle=control
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=F bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=D bundle=control
#pragma HLS INTERFACE s_axilite port=G bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int ni = NI;
    const int nj = NJ;
    const int nk = NK;
    const int nl = NL;
    const int nm = NM;

  int i, j, k, ti;

  // Full-matrix local staging buffers (shared working set)
  static double l_A[NI][NK];
  static double l_B[NK][NJ];
  static double l_C[NJ][NM];
  static double l_D[NM][NL];
  static double l_E[NI][NJ];
  static double l_F[NJ][NL];
  static double l_G[NI][NL];

  // Partition along the reduction dimension so the K loop can read
  // multiple elements per cycle when the J loop is pipelined.
#pragma HLS ARRAY_PARTITION variable=l_A cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_B cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_C cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_D cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_E cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_F cyclic factor=8 dim=1

  // ------------------------------------------------------------------
  // LOAD phase: bring all required inputs from global memory on-chip.
  // ------------------------------------------------------------------
  LOAD_A_I: for (i = 0; i < ni; i++)
    LOAD_A_K: for (k = 0; k < nk; k++) {
#pragma HLS PIPELINE II=1
      l_A[i][k] = A[i][k];
    }

  LOAD_B_K: for (k = 0; k < nk; k++)
    LOAD_B_J: for (j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
      l_B[k][j] = B[k][j];
    }

  LOAD_C_I: for (i = 0; i < nj; i++)
    LOAD_C_K: for (k = 0; k < nm; k++) {
#pragma HLS PIPELINE II=1
      l_C[i][k] = C[i][k];
    }

  LOAD_D_K: for (k = 0; k < nm; k++)
    LOAD_D_J: for (j = 0; j < nl; j++) {
#pragma HLS PIPELINE II=1
      l_D[k][j] = D[k][j];
    }

  // ------------------------------------------------------------------
  // COMPUTE phase: operate purely on local buffers, organized in tiles
  // of output rows so each tile is a bounded working set.
  // ------------------------------------------------------------------

  // E = A * B  (process NI rows in tiles of TILE rows)
  E_TILE: for (ti = 0; ti < ni; ti += TILE) {
    int i_end = ti + TILE; if (i_end > ni) i_end = ni;
    E_I: for (i = ti; i < i_end; i++)
      E_J: for (j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
        double acc = 0.0;
        E_K: for (k = 0; k < nk; ++k) {
#pragma HLS UNROLL factor=8
          acc += l_A[i][k] * l_B[k][j];
        }
        l_E[i][j] = acc;
      }
  }

  // F = C * D  (process NJ rows in tiles of TILE rows)
  F_TILE: for (ti = 0; ti < nj; ti += TILE) {
    int i_end = ti + TILE; if (i_end > nj) i_end = nj;
    F_I: for (i = ti; i < i_end; i++)
      F_J: for (j = 0; j < nl; j++) {
#pragma HLS PIPELINE II=1
        double acc = 0.0;
        F_K: for (k = 0; k < nm; ++k) {
#pragma HLS UNROLL factor=8
          acc += l_C[i][k] * l_D[k][j];
        }
        l_F[i][j] = acc;
      }
  }

  // G = E * F  (process NI rows in tiles of TILE rows)
  G_TILE: for (ti = 0; ti < ni; ti += TILE) {
    int i_end = ti + TILE; if (i_end > ni) i_end = ni;
    G_I: for (i = ti; i < i_end; i++)
      G_J: for (j = 0; j < nl; j++) {
#pragma HLS PIPELINE II=1
        double acc = 0.0;
        G_K: for (k = 0; k < nj; ++k) {
#pragma HLS UNROLL factor=8
          acc += l_E[i][k] * l_F[k][j];
        }
        l_G[i][j] = acc;
      }
  }

  // ------------------------------------------------------------------
  // STORE phase: write all results back to global memory.
  // ------------------------------------------------------------------
  STORE_E_I: for (i = 0; i < ni; i++)
    STORE_E_J: for (j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
      E[i][j] = l_E[i][j];
    }

  STORE_F_I: for (i = 0; i < nj; i++)
    STORE_F_J: for (j = 0; j < nl; j++) {
#pragma HLS PIPELINE II=1
      F[i][j] = l_F[i][j];
    }

  STORE_G_I: for (i = 0; i < ni; i++)
    STORE_G_J: for (j = 0; j < nl; j++) {
#pragma HLS PIPELINE II=1
      G[i][j] = l_G[i][j];
    }
}