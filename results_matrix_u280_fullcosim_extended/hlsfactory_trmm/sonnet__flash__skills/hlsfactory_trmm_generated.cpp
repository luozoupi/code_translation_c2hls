#include "trmm.h"

extern "C" {

void kernel_trmm(
        double alpha,
        double A[M + 0][M + 0],
        double B[M + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=alpha bundle=control
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local buffers to avoid repeated global memory access
    double localA[M][M];
    double localB[M][N];

#pragma HLS ARRAY_PARTITION variable=localA cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=localB cyclic factor=4 dim=2

    // Load A into local buffer
    load_A_i: for (int i = 0; i < M; i++) {
        load_A_j: for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
            localA[i][j] = A[i][j];
        }
    }

    // Load B into local buffer
    load_B_i: for (int i = 0; i < M; i++) {
        load_B_j: for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            localB[i][j] = B[i][j];
        }
    }

    // Compute TRMM
    // Preserve original algorithm exactly; pipeline innermost k-loop.
    // The FP reduction in k is serial to maintain bit-exactness.
    const int m = M;
    const int n = N;

    int i, j, k;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            // Serial FP reduction: do NOT unroll or reorder (bit-exact guard)
            for (k = i + 1; k < m; k++) {
#pragma HLS PIPELINE
                localB[i][j] += localA[k][i] * localB[k][j];
            }
            localB[i][j] = alpha * localB[i][j];
        }
    }

    // Store B back to global memory
    store_B_i: for (int i = 0; i < M; i++) {
        store_B_j: for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            B[i][j] = localB[i][j];
        }
    }
}

} // extern "C"