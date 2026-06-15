#include "bicg.h"

extern "C" {

void kernel_bicg(
        double A[N + 0][M + 0],
        double s[M + 0],
        double q[N + 0],
        double p[M + 0],
        double r[N + 0])
{
#pragma HLS INTERFACE m_axi port=A      offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=s      offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=q      offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=p      offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=r      offset=slave bundle=gmem4
#pragma HLS INTERFACE s_axilite port=A      bundle=control
#pragma HLS INTERFACE s_axilite port=s      bundle=control
#pragma HLS INTERFACE s_axilite port=q      bundle=control
#pragma HLS INTERFACE s_axilite port=p      bundle=control
#pragma HLS INTERFACE s_axilite port=r      bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local buffers for parallel access and banking
    double l_s[M];
    double l_q[N];
    double l_p[M];
    double l_r[N];
    double l_A[N][M];

#pragma HLS ARRAY_PARTITION variable=l_s cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_q cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_p cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_r cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_A cyclic factor=8 dim=2

    // Load inputs from global memory
    load_r: for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        l_r[i] = r[i];
    }

    load_p: for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
        l_p[j] = p[j];
    }

    load_A: for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
            l_A[i][j] = A[i][j];
        }
    }

    // Initialize s
    init_s: for (int i = 0; i < M; i++) {
#pragma HLS PIPELINE II=1
        l_s[i] = 0.0;
    }

    // Main compute loops
    compute: for (int i = 0; i < N; i++) {
        double q_acc = 0.0;
        double ri = l_r[i];

        inner: for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
            double aij = l_A[i][j];
            l_s[j] = l_s[j] + ri * aij;
            q_acc   = q_acc  + aij * l_p[j];
        }

        l_q[i] = q_acc;
    }

    // Write outputs back to global memory
    store_s: for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
        s[j] = l_s[j];
    }

    store_q: for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        q[i] = l_q[i];
    }
}

} // extern "C"