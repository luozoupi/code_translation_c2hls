#include "lu.h"
#include <cstring>

extern "C" {
void kernel_lu(
	       double A[ N + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;

  int i, j, k;

  // Local buffers for tiling:
  //  - row_i: current working row A[i][*]
  //  - tile : staged pivot rows A[0..i-1][*] reused across the j/k loops
  static double row_i[N];
  static double tile[N][N];
#pragma HLS ARRAY_PARTITION variable=row_i cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=tile cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=tile cyclic factor=4 dim=2

  for (i = 0; i < n; i++) {

    // ---------- LOAD phase ----------
    // Bring current row A[i][*] into local buffer
  LOAD_ROW:
    for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
      row_i[j] = A[i][j];
    }
    // Bring the newly finalized previous row A[i-1][*] into the tile cache
    if (i > 0) {
    LOAD_TILE:
      for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
        tile[i - 1][j] = A[i - 1][j];
      }
    }

    // ---------- COMPUTE phase (on local buffers) ----------
  COMPUTE_LOWER:
    for (j = 0; j < i; j++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=N
      double acc = row_i[j];
    COMPUTE_LOWER_K:
      for (k = 0; k < j; k++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=N
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=2
        acc -= row_i[k] * tile[k][j];
      }
      row_i[j] = acc / tile[j][j];
    }
  COMPUTE_UPPER:
    for (j = i; j < n; j++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=N
      double acc = row_i[j];
    COMPUTE_UPPER_K:
      for (k = 0; k < i; k++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=N
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=2
        acc -= row_i[k] * tile[k][j];
      }
      row_i[j] = acc;
    }

    // ---------- STORE phase ----------
  STORE_ROW:
    for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
      A[i][j] = row_i[j];
    }
  }
}
}