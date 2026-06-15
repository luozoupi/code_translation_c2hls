#include "bicg.h"

extern "C" {

void kernel_bicg(
        double A[N + 0][M + 0],
        double s[M + 0],
        double q[N + 0],
        double p[M + 0],
        double r[N + 0])
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

    // Local buffers to avoid repeated global memory reads and enable pipelining
    double s_local[M];
    double q_local[N];
    double p_local[M];
    double r_local[N];

#pragma HLS ARRAY_PARTITION variable=s_local cyclic factor=1 dim=1
#pragma HLS ARRAY_PARTITION variable=p_local cyclic factor=1 dim=1

    // Load p and r into local buffers
    load_p: for (int j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
        p_local[j] = p[j];
    }

    load_r: for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        r_local[i] = r[i];
    }

    // Initialize s_local
    init_s: for (int i = 0; i < m; i++) {
#pragma HLS PIPELINE II=1
        s_local[i] = 0.0;
    }

    // Initialize q_local
    init_q: for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        q_local[i] = 0.0;
    }

    // Main computation
    // Note: s[j] is accumulated across outer i-loop iterations (loop-carried dep across i),
    // so we keep the outer loop serial. The inner j-loop pipelines safely.
    // q[i] is a reduction over j within each i-iteration — keep serial (FP guard).
    outer: for (int i = 0; i < n; i++) {
        double qi = q_local[i]; // scalar accumulator for q[i]
        double ri = r_local[i];
        inner: for (int j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
            double aij = A[i][j];
            s_local[j] = s_local[j] + ri * aij;
            qi = qi + aij * p_local[j];
        }
        q_local[i] = qi;
    }

    // Write s_local back to global
    store_s: for (int j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
        s[j] = s_local[j];
    }

    // Write q_local back to global
    store_q: for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        q[i] = q_local[i];
    }
}

} // extern "C"