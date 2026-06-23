#include "bicg.h"
#include <cstring>

#define TILE_SIZE 256

void kernel_bicg( 
		 double A[ N + 0][M + 0],
		 double s[ M + 0],
		 double q[ N + 0],
		 double p[ M + 0],
		 double r[ N + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=s offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=q offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=p offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=r offset=slave bundle=gmem4
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=s bundle=control
#pragma HLS INTERFACE s_axilite port=q bundle=control
#pragma HLS INTERFACE s_axilite port=p bundle=control
#pragma HLS INTERFACE s_axilite port=r bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;
    const int m = M;

  int i, j, jt;

  // Local buffers for the full M-dimension working set (s and p reused across all rows)
  double s_local[M];
  double p_local[M];

  // Local buffer for one tile of an A row
  double A_tile[TILE_SIZE];

  // Local buffers for q and r (load r, accumulate q)
  double q_local[N];
  double r_local[N];

  // ---------- LOAD phase: p and r into local buffers, init s_local ----------
  load_p:
  for (j = 0; j < m; j++) {
#pragma HLS LOOP_TRIPCOUNT min=116 max=116
#pragma HLS PIPELINE II=1
    p_local[j] = p[j];
    s_local[j] = 0.0;
  }

  load_r:
  for (i = 0; i < n; i++) {
#pragma HLS LOOP_TRIPCOUNT min=124 max=124
#pragma HLS PIPELINE II=1
    r_local[i] = r[i];
  }

  // ---------- COMPUTE phase: tiled over M ----------
  outer_n:
  for (i = 0; i < n; i++)
    {
#pragma HLS LOOP_TRIPCOUNT min=124 max=124
      double q_acc = 0.0;
      double r_i = r_local[i];

      // Process the M dimension in tiles
      tile_loop:
      for (jt = 0; jt < m; jt += TILE_SIZE)
        {
#pragma HLS LOOP_TRIPCOUNT min=1 max=1
          int tile_end = jt + TILE_SIZE;
          if (tile_end > m) tile_end = m;
          int tile_len = tile_end - jt;

          // ----- LOAD tile of A row into local buffer -----
          load_tile:
          for (j = 0; j < tile_len; j++) {
#pragma HLS LOOP_TRIPCOUNT min=116 max=116
#pragma HLS PIPELINE II=1
            A_tile[j] = A[i][jt + j];
          }

          // ----- COMPUTE on the local tile -----
          compute_tile:
          for (j = 0; j < tile_len; j++) {
#pragma HLS LOOP_TRIPCOUNT min=116 max=116
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=s_local inter false
            double a = A_tile[j];
            s_local[jt + j] = s_local[jt + j] + r_i * a;
            q_acc = q_acc + a * p_local[jt + j];
          }
        }

      q_local[i] = q_acc;
    }

  // ---------- STORE phase: write back q and s ----------
  store_q:
  for (i = 0; i < n; i++) {
#pragma HLS LOOP_TRIPCOUNT min=124 max=124
#pragma HLS PIPELINE II=1
    q[i] = q_local[i];
  }

  store_s:
  for (i = 0; i < m; i++) {
#pragma HLS LOOP_TRIPCOUNT min=116 max=116
#pragma HLS PIPELINE II=1
    s[i] = s_local[i];
  }

}