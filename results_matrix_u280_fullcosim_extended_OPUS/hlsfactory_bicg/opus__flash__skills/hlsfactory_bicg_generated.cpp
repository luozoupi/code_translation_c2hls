#include "bicg.h"


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

  // Local buffers to enable banking / fast on-chip access
  double s_local[M];
  double p_local[M];
  double r_local[N];
#pragma HLS ARRAY_PARTITION variable=s_local cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=p_local cyclic factor=8 dim=1

  // Stage p into local memory (reused across all i iterations)
  load_p: for (j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
    p_local[j] = p[j];
  }

  // Stage r into local memory
  load_r: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
    r_local[i] = r[i];
  }

  // Initialize s accumulator
  init_s: for (i = 0; i < m; i++) {
#pragma HLS PIPELINE II=1
    s_local[i] = 0;
  }

  main_i: for (i = 0; i < n; i++)
    {
      double q_acc = 0.0;
      double ri = r_local[i];

      inner_j: for (j = 0; j < m; j++)
	{
#pragma HLS PIPELINE II=1
	  double a = A[i][j];
	  s_local[j] = s_local[j] + ri * a;        // independent across j
	  q_acc = q_acc + a * p_local[j];          // serial reduction, order preserved
	}

      q[i] = q_acc;
    }

  // Write back s accumulator
  store_s: for (i = 0; i < m; i++) {
#pragma HLS PIPELINE II=1
    s[i] = s_local[i];
  }

}