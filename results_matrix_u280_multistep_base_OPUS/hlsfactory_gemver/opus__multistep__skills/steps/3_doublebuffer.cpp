#include "gemver.h"
#include <string.h>

#define TILE 8

// Load a tile of A rows from global memory into one of the ping-pong buffers.
static void load_tile(double A[N + 0][N + 0],
                      double buf0[TILE][N], double buf1[TILE][N],
                      int ti, int rows, int flag) {
  if (flag == 0) {
    load0: for (int r = 0; r < rows; r++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
      memcpy(&buf0[r][0], &A[ti + r][0], N * sizeof(double));
    }
  } else {
    load1: for (int r = 0; r < rows; r++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
      memcpy(&buf1[r][0], &A[ti + r][0], N * sizeof(double));
    }
  }
}

// Compute: copy staged tile rows into the full local matrix l_A.
static void compute_tile(double l_A[N][N],
                         double buf0[TILE][N], double buf1[TILE][N],
                         int ti, int rows, int flag) {
  if (flag == 0) {
    comp0: for (int r = 0; r < rows; r++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
      copy0: for (int c = 0; c < N; c++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
        l_A[ti + r][c] = buf0[r][c];
      }
    }
  } else {
    comp1: for (int r = 0; r < rows; r++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
      copy1: for (int c = 0; c < N; c++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
        l_A[ti + r][c] = buf1[r][c];
      }
    }
  }
}

void kernel_gemver(
		   double alpha,
		   double beta,
		   double A[ N + 0][N + 0],
		   double u1[ N + 0],
		   double v1[ N + 0],
		   double u2[ N + 0],
		   double v2[ N + 0],
		   double w[ N + 0],
		   double x[ N + 0],
		   double y[ N + 0],
		   double z[ N + 0])
{
#pragma HLS INTERFACE m_axi port=A   offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=u1  offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=v1  offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=u2  offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=v2  offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=w   offset=slave bundle=gmem5
#pragma HLS INTERFACE m_axi port=x   offset=slave bundle=gmem6
#pragma HLS INTERFACE m_axi port=y   offset=slave bundle=gmem7
#pragma HLS INTERFACE m_axi port=z   offset=slave bundle=gmem8

#pragma HLS INTERFACE s_axilite port=alpha bundle=control
#pragma HLS INTERFACE s_axilite port=beta  bundle=control
#pragma HLS INTERFACE s_axilite port=A     bundle=control
#pragma HLS INTERFACE s_axilite port=u1    bundle=control
#pragma HLS INTERFACE s_axilite port=v1    bundle=control
#pragma HLS INTERFACE s_axilite port=u2    bundle=control
#pragma HLS INTERFACE s_axilite port=v2    bundle=control
#pragma HLS INTERFACE s_axilite port=w     bundle=control
#pragma HLS INTERFACE s_axilite port=x     bundle=control
#pragma HLS INTERFACE s_axilite port=y     bundle=control
#pragma HLS INTERFACE s_axilite port=z     bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;

  int i, j, ti;

  // Stage vectors into local buffers for fast reuse across the loop nests.
  double l_u1[N], l_v1[N], l_u2[N], l_v2[N];
  double l_x[N], l_y[N], l_z[N], l_w[N];
#pragma HLS ARRAY_PARTITION variable=l_v1 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_v2 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_x  cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_y  cyclic factor=8 dim=1

  // Local tile/buffer for the full matrix A (staged once, reused across phases).
  static double l_A[N][N];
#pragma HLS ARRAY_PARTITION variable=l_A cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_A cyclic factor=8 dim=1

  // ---- Ping-pong tile buffers for double-buffered staging of A ----
  static double tileA_0[TILE][N];
  static double tileA_1[TILE][N];
#pragma HLS ARRAY_PARTITION variable=tileA_0 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=tileA_1 cyclic factor=8 dim=2

  load_vecs: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
    l_u1[i] = u1[i];
    l_v1[i] = v1[i];
    l_u2[i] = u2[i];
    l_v2[i] = v2[i];
    l_x[i]  = x[i];
    l_y[i]  = y[i];
    l_z[i]  = z[i];
    l_w[i]  = w[i];
  }

  // ---- DOUBLE-BUFFERED LOAD PHASE: stage matrix A tile by tile ----
  // Load of tile k+1 overlaps with the staging (compute) of tile k.
  int num_tiles = (n + TILE - 1) / TILE;

  // Prologue: load first tile.
  {
    int rows0 = (TILE <= n) ? TILE : n;
    load_tile(A, tileA_0, tileA_1, 0, rows0, 0);
  }

  stage_A: for (int t = 0; t < num_tiles; t++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=15
    int ti_cur  = t * TILE;
    int rows_cur = (ti_cur + TILE <= n) ? TILE : (n - ti_cur);
    int flag_cur = t & 1;

    // Load next tile into the OTHER buffer (overlaps with compute of current).
    if (t + 1 < num_tiles) {
      int ti_next  = (t + 1) * TILE;
      int rows_next = (ti_next + TILE <= n) ? TILE : (n - ti_next);
      load_tile(A, tileA_0, tileA_1, ti_next, rows_next, (t + 1) & 1);
    }

    // Compute (stage current tile into full l_A) from current buffer.
    compute_tile(l_A, tileA_0, tileA_1, ti_cur, rows_cur, flag_cur);
  }

  // ---- COMPUTE PHASE (operates entirely on local buffers) ----

  // Phase 1: A = A + u1*v1^T + u2*v2^T
  phase1_i: for (i = 0; i < n; i++) {
    double u1i = l_u1[i];
    double u2i = l_u2[i];
    phase1_j: for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS DEPENDENCE variable=l_A inter false
      l_A[i][j] = l_A[i][j] + u1i * l_v1[j] + u2i * l_v2[j];
    }
  }

  // Phase 2: x = x + beta * A^T * y
  phase2_i: for (i = 0; i < n; i++) {
    double acc = l_x[i];
    phase2_j: for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS DEPENDENCE variable=l_A inter false
      acc += beta * l_A[j][i] * l_y[j];
    }
    l_x[i] = acc;
  }

  // Phase 3: x = x + z
  phase3: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
    l_x[i] = l_x[i] + l_z[i];
  }

  // Phase 4: w = w + alpha * A * x
  phase4_i: for (i = 0; i < n; i++) {
    double acc = l_w[i];
    phase4_j: for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS DEPENDENCE variable=l_A inter false
      acc += alpha * l_A[i][j] * l_x[j];
    }
    l_w[i] = acc;
  }

  // ---- STORE PHASE: write back modified matrix A (tile by tile) and result vectors ----
  const int STILE = 256;
  store_A_outer: for (ti = 0; ti < n; ti += STILE) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=1
    int rows = (ti + STILE <= n) ? STILE : (n - ti);
    store_A_rows: for (int r = 0; r < rows; r++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=120
      memcpy(&A[ti + r][0], &l_A[ti + r][0], n * sizeof(double));
    }
  }

  store_out: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
    x[i] = l_x[i];
    w[i] = l_w[i];
  }
}