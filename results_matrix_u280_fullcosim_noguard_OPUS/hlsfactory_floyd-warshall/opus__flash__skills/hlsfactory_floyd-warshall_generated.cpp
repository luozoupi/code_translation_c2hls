#include "floyd-warshall.h"


void kernel_floyd_warshall(
			   int path[ N + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=path offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=path bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;

  int i, j, k;

  // Local buffers to cache the k-th row and k-th column for the current k.
  // This removes repeated reads of path[k][j] and path[i][k] from the
  // pipelined inner loop, avoiding memory-port conflicts.
  int row_k[N];
  int col_k[N];
#pragma HLS ARRAY_PARTITION variable=row_k cyclic factor=8 dim=1

  for (k = 0; k < n; k++)
    {
      // Stage the k-th row and k-th column into local memory.
    cache_k:
      for (int t = 0; t < n; t++)
	{
#pragma HLS PIPELINE II=1
	  row_k[t] = path[k][t];
	  col_k[t] = path[t][k];
	}

      for(i = 0; i < n; i++)
	{
	  int pik = col_k[i];
	row_loop:
	  for (j = 0; j < n; j++)
	    {
#pragma HLS PIPELINE II=1
	      int via = pik + row_k[j];
	      int cur = path[i][j];
	      path[i][j] = cur < via ? cur : via;
	    }
	}
    }

}