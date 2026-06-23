#include "bicg.h"
#include <cstring>

#define TILE_SIZE 256

static void load_A_row(double A[N + 0][M + 0],
                       double A_tile_1[TILE_SIZE],
                       double A_tile_2[TILE_SIZE],
                       int i, int m, int flag)
{
  load_tile:
  for (int j = 0; j < m; j++) {
#pragma HLS LOOP_TRIPCOUNT min=116 max=116
#pragma HLS PIPELINE II=1
    if (flag == 0)
      A_tile_1[j] = A[i][j];
    else
      A_tile_2[j] = A[i][j];
  }
}

static void compute_A_row(double A_tile_1[TILE_SIZE],
                          double A_tile_2[TILE_SIZE],
                          double s_local[M],
                          double p_local[M],
                          double q_local[N],
                          double r_i, int i, int m, int flag)
{
  double q_acc = 0.0;
  compute_tile:
  for (int j = 0; j < m; j++) {
#pragma HLS LOOP_TRIPCOUNT min=116 max=116
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS DEPENDENCE variable=s_local inter false
    double a = (flag == 0) ? A_tile_1[j] : A_tile_2[j];
    s_local[j] = s_local[j] + r_i * a;
    q_acc = q_acc + a * p_local[j];
  }
  q_local[i] = q_acc;
}

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

  int i, j;

  // Local buffers for the full M-dimension working set (s and p reused across all rows)
  double s_local[M];
#pragma HLS ARRAY_PARTITION variable=s_local cyclic factor=4 dim=1
  double p_local[M];
#pragma HLS ARRAY_PARTITION variable=p_local cyclic factor=4 dim=1

  // Double-buffered local buffers for one tile of an A row
  double A_tile_1[TILE_SIZE];
#pragma HLS ARRAY_PARTITION variable=A_tile_1 cyclic factor=4 dim=1
  double A_tile_2[TILE_SIZE];
#pragma HLS ARRAY_PARTITION variable=A_tile_2 cyclic factor=4 dim=1

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

  // ---------- COMPUTE phase: double-buffered over rows ----------
  // Prologue: load first row's tile into buffer 0
  load_A_row(A, A_tile_1, A_tile_2, 0, m, 0);

  outer_n:
  for (i = 0; i < n; i++)
    {
#pragma HLS LOOP_TRIPCOUNT min=124 max=124
      int flag = i % 2;  // buffer holding the row currently being computed

      // Load next row's tile into the OTHER buffer while we compute this one
      if (i + 1 < n) {
        load_A_row(A, A_tile_1, A_tile_2, i + 1, m, (i + 1) % 2);
      }

      // Compute on the current row's tile
      compute_A_row(A_tile_1, A_tile_2, s_local, p_local, q_local,
                    r_local[i], i, m, flag);
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