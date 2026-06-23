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

  for (k = 0; k < n; k++)
    {
      for(i = 0; i < n; i++)
	for (j = 0; j < n; j++)
	  {
#pragma HLS PIPELINE II=1
	  path[i][j] = path[i][j] < path[i][k] + path[k][j] ?
	    path[i][j] : path[i][k] + path[k][j];
	  }
    }

}
}