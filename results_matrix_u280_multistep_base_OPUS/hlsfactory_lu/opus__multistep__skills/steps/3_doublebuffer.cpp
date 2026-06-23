#include "lu.h"
#include <cstring>

extern "C" {

// Load row A[i][*] into the selected row buffer
static void load_row(double A[N + 0][N + 0],
                     double row_i_0[N], double row_i_1[N],
                     int i, int flag)
{
  const int n = N;
  if (flag == 0) {
  LOAD_ROW_0:
    for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
      row_i_0[j] = A[i][j];
    }
  } else {
  LOAD_ROW_1:
    for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
      row_i_1[j] = A[i][j];
    }
  }
}

// Compute on selected row buffer, using shared tile cache
static void compute_row(double row_i_0[N], double row_i_1[N],
                        double tile[N][N], int i, int flag)
{
  const int n = N;
  int j, k;

  if (flag == 0) {
  COMPUTE_LOWER_0:
    for (j = 0; j < i; j++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=N
      double acc = row_i_0[j];
    COMPUTE_LOWER_K_0:
      for (k = 0; k < j; k++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=N
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=2
        acc -= row_i_0[k] * tile[k][j];
      }
      row_i_0[j] = acc / tile[j][j];
    }
  COMPUTE_UPPER_0:
    for (j = i; j < n; j++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=N
      double acc = row_i_0[j];
    COMPUTE_UPPER_K_0:
      for (k = 0; k < i; k++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=N
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=2
        acc -= row_i_0[k] * tile[k][j];
      }
      row_i_0[j] = acc;
    }
  } else {
  COMPUTE_LOWER_1:
    for (j = 0; j < i; j++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=N
      double acc = row_i_1[j];
    COMPUTE_LOWER_K_1:
      for (k = 0; k < j; k++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=N
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=2
        acc -= row_i_1[k] * tile[k][j];
      }
      row_i_1[j] = acc / tile[j][j];
    }
  COMPUTE_UPPER_1:
    for (j = i; j < n; j++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=N
      double acc = row_i_1[j];
    COMPUTE_UPPER_K_1:
      for (k = 0; k < i; k++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=N
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=2
        acc -= row_i_1[k] * tile[k][j];
      }
      row_i_1[j] = acc;
    }
  }
}

// Store selected row buffer back to A[i][*]
static void store_row(double A[N + 0][N + 0],
                      double row_i_0[N], double row_i_1[N],
                      int i, int flag)
{
  const int n = N;
  if (flag == 0) {
  STORE_ROW_0:
    for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
      A[i][j] = row_i_0[j];
    }
  } else {
  STORE_ROW_1:
    for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
      A[i][j] = row_i_1[j];
    }
  }
}

void kernel_lu(
	       double A[ N + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;

  int i, j;

  // Double-buffered working row: ping-pong between two copies
  static double row_i_0[N];
  static double row_i_1[N];
  // Shared tile cache holding finalized previous rows
  static double tile[N][N];
#pragma HLS ARRAY_PARTITION variable=row_i_0 cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=row_i_1 cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=tile cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=tile cyclic factor=4 dim=2

  // Prologue: load first row into buffer 0
  load_row(A, row_i_0, row_i_1, 0, 0);

  for (i = 0; i < n; i++) {

    int flag = i % 2;          // buffer currently being computed/stored
    int next_flag = (i + 1) % 2; // buffer used to prefetch next row

    // ---------- LOAD phase ----------
    // Prefetch next row A[i+1][*] into the OTHER buffer (overlaps with compute)
    if (i + 1 < n) {
      load_row(A, row_i_0, row_i_1, i + 1, next_flag);
    }

    // Bring the newly finalized previous row A[i-1][*] into the tile cache
    if (i > 0) {
    LOAD_TILE:
      for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
        tile[i - 1][j] = A[i - 1][j];
      }
    }

    // ---------- COMPUTE phase (on current buffer) ----------
    compute_row(row_i_0, row_i_1, tile, i, flag);

    // ---------- STORE phase ----------
    store_row(A, row_i_0, row_i_1, i, flag);
  }
}
}