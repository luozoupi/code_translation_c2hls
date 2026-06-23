#include "floyd-warshall.h"


void kernel_floyd_warshall(
			   int path[ N + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=path offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=path bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;

  int i, j, k;

  // Local buffer to enable fast on-chip access and parallel ports
  static int buf[N][N];
#pragma HLS ARRAY_PARTITION variable=buf cyclic factor=16 dim=2

  // Stage data into local memory
  load_i: for (i = 0; i < n; i++) {
    load_j: for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
      buf[i][j] = path[i][j];
    }
  }

  k_loop: for (k = 0; k < n; k++)
    {
      i_loop: for (i = 0; i < n; i++) {
	// Cache path[i][k] which is invariant across the j loop
	int pik = buf[i][k];
	j_loop: for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=16
	  int pij = buf[i][j];
	  int pkj = buf[k][j];
	  int sum = pik + pkj;
	  buf[i][j] = pij < sum ? pij : sum;
	}
      }
    }

  // Write results back to global memory
  store_i: for (i = 0; i < n; i++) {
    store_j: for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
      path[i][j] = buf[i][j];
    }
  }

}