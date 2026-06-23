#include "3mm.h"
#include <string.h>

#define TILE 8

// Load a tile of A rows into one of the two ping-pong buffers.
static void load_A_tile(double A[NI][NK],
                        double buf1[TILE][NK],
                        double buf2[TILE][NK],
                        int ti, int i_end, int flag)
{
  if (flag == 0) {
    LD_A1_I: for (int i = ti; i < i_end; i++)
      LD_A1_K: for (int k = 0; k < NK; k++) {
#pragma HLS PIPELINE II=1
        buf1[i - ti][k] = A[i][k];
      }
  } else {
    LD_A2_I: for (int i = ti; i < i_end; i++)
      LD_A2_K: for (int k = 0; k < NK; k++) {
#pragma HLS PIPELINE II=1
        buf2[i - ti][k] = A[i][k];
      }
  }
}

// Compute E rows for a tile from one of the two ping-pong buffers.
static void compute_E_tile(double buf1[TILE][NK],
                           double buf2[TILE][NK],
                           double l_B[NK][NJ],
                           double l_E[NI][NJ],
                           int ti, int i_end, int flag)
{
  if (flag == 0) {
    CE1_I: for (int i = ti; i < i_end; i++)
      CE1_J: for (int j = 0; j < NJ; j++) {
#pragma HLS PIPELINE II=1
        double acc = 0.0;
        CE1_K: for (int k = 0; k < NK; ++k) {
#pragma HLS UNROLL factor=8
          acc += buf1[i - ti][k] * l_B[k][j];
        }
        l_E[i][j] = acc;
      }
  } else {
    CE2_I: for (int i = ti; i < i_end; i++)
      CE2_J: for (int j = 0; j < NJ; j++) {
#pragma HLS PIPELINE II=1
        double acc = 0.0;
        CE2_K: for (int k = 0; k < NK; ++k) {
#pragma HLS UNROLL factor=8
          acc += buf2[i - ti][k] * l_B[k][j];
        }
        l_E[i][j] = acc;
      }
  }
}

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
  static double l_B[NK][NJ];
  static double l_C[NJ][NM];
  static double l_D[NM][NL];
  static double l_E[NI][NJ];
  static double l_F[NJ][NL];
  static double l_G[NI][NL];

  // Ping-pong buffers for tiles of A rows.
  static double l_A_buf1[TILE][NK];
  static double l_A_buf2[TILE][NK];

  // Partition along the reduction dimension so the K loop can read
  // multiple elements per cycle when the J loop is pipelined.
#pragma HLS ARRAY_PARTITION variable=l_A_buf1 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_A_buf2 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_B cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_C cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_D cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_E cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_F cyclic factor=8 dim=1

  // ------------------------------------------------------------------
  // LOAD phase: bring B, C, D inputs from global memory on-chip.
  // (A is now loaded tile-by-tile in the double-buffered E compute.)
  // ------------------------------------------------------------------
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
  // COMPUTE E = A * B  with DOUBLE BUFFERING over tiles of A rows.
  //   - Load tile k+1 of A while computing tile k of E.
  //   - flag alternates each tile to select which buffer set is used.
  // ------------------------------------------------------------------
  {
    // Number of tiles over the NI dimension.
    int num_tiles = (ni + TILE - 1) / TILE;

    // Prologue: load the first tile into buffer set 0.
    if (num_tiles > 0) {
      int ti0 = 0;
      int i_end0 = ti0 + TILE; if (i_end0 > ni) i_end0 = ni;
      load_A_tile(A, l_A_buf1, l_A_buf2, ti0, i_end0, 0);
    }

    // Steady state: for each tile, compute current tile from one buffer
    // while loading the next tile into the other buffer.
    E_DB_TILE: for (int t = 0; t < num_tiles; t++) {
      int ti_cur = t * TILE;
      int i_end_cur = ti_cur + TILE; if (i_end_cur > ni) i_end_cur = ni;
      int flag_cur = t & 1;

      // Pre-load next tile (if any) into the opposite buffer set.
      int t_next = t + 1;
      if (t_next < num_tiles) {
        int ti_next = t_next * TILE;
        int i_end_next = ti_next + TILE; if (i_end_next > ni) i_end_next = ni;
        int flag_next = t_next & 1;
        load_A_tile(A, l_A_buf1, l_A_buf2, ti_next, i_end_next, flag_next);
      }

      // Compute current tile from the buffer that was loaded earlier.
      compute_E_tile(l_A_buf1, l_A_buf2, l_B, l_E, ti_cur, i_end_cur, flag_cur);
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