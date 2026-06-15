#include "trmm.h"

extern "C" {

void kernel_trmm(
        double alpha,
        double A[M + 0][M + 0],
        double B[M + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=3600
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=4800
#pragma HLS INTERFACE s_axilite port=alpha bundle=control
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local buffers to avoid repeated global memory access and enable partitioning
    double A_local[M][M];
    double B_local[M][N];

#pragma HLS ARRAY_PARTITION variable=A_local cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=B_local cyclic factor=8 dim=2

    // Load A into local buffer
    load_A_outer: for (int i = 0; i < M; i++) {
        load_A_inner: for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
            A_local[i][j] = A[i][j];
        }
    }

    // Load B into local buffer
    load_B_outer: for (int i = 0; i < M; i++) {
        load_B_inner: for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            B_local[i][j] = B[i][j];
        }
    }

    // Compute: preserve original algorithm exactly
    const int m = M;
    const int n = N;

    int i, j, k;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            double b_val = B_local[i][j];
            for (k = i + 1; k < m; k++) {
#pragma HLS PIPELINE II=1
                b_val += A_local[k][i] * B_local[k][j];
            }
            B_local[i][j] = alpha * b_val;
        }
    }

    // Store B back to global memory
    store_B_outer: for (int i = 0; i < M; i++) {
        store_B_inner: for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            B[i][j] = B_local[i][j];
        }
    }
}

} // extern "C"