#include "floyd-warshall.h"


extern "C" {
void kernel_floyd_warshall(
			   int path[ N + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=path offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=path bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;

  int i, j, k;

  // Local buffer to stage the matrix for reuse during the k-iterations.
  // This enables fast, partitioned on-chip access for the inner loop.
  static int local_path[N][N];
#pragma HLS ARRAY_PARTITION variable=local_path cyclic factor=16 dim=2

  // Load from global memory into local buffer
  load_rows:
  for (i = 0; i < n; i++) {
    load_cols:
    for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
      local_path[i][j] = path[i][j];
    }
  }

  // Core Floyd-Warshall on the local buffer
  k_loop:
  for (k = 0; k < n; k++)
    {
      i_loop:
      for(i = 0; i < n; i++) {
	j_loop:
	for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
	  int via = local_path[i][k] + local_path[k][j];
	  int cur = local_path[i][j];
	  local_path[i][j] = cur < via ? cur : via;
	}
      }
    }

  // Store back to global memory
  store_rows:
  for (i = 0; i < n; i++) {
    store_cols:
    for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
      path[i][j] = local_path[i][j];
    }
  }

}
}