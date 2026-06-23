#include "bicg.h"

extern "C" {
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

    // Local staging buffers for reuse and parallel access
    double s_local[M];
    double p_local[M];
    double r_local[N];
    double q_local[N];
#pragma HLS ARRAY_PARTITION variable=s_local cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=p_local cyclic factor=8 dim=1

    // Load p into local buffer
  load_p:
    for (j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
        p_local[j] = p[j];
    }

    // Load r into local buffer
  load_r:
    for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        r_local[i] = r[i];
    }

    // Initialize s
  init_s:
    for (i = 0; i < m; i++) {
#pragma HLS PIPELINE II=1
        s_local[i] = 0.0;
    }

  outer:
    for (i = 0; i < n; i++)
    {
        double q_acc = 0.0;
        double r_i = r_local[i];
      inner:
        for (j = 0; j < m; j++)
        {
#pragma HLS PIPELINE II=1
            double a = A[i][j];
            s_local[j] = s_local[j] + r_i * a;
            q_acc = q_acc + a * p_local[j];
        }
        q_local[i] = q_acc;
    }

    // Write back results
  store_s:
    for (j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
        s[j] = s_local[j];
    }

  store_q:
    for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        q[i] = q_local[i];
    }
}
}