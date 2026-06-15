#include "syr2k.h"

extern "C" {

void kernel_syr2k(
        double alpha,
        double beta,
        double C[N + 0][N + 0],
        double A[N + 0][M + 0],
        double B[N + 0][M + 0])
{
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=alpha bundle=control
#pragma HLS INTERFACE s_axilite port=beta  bundle=control
#pragma HLS INTERFACE s_axilite port=C     bundle=control
#pragma HLS INTERFACE s_axilite port=A     bundle=control
#pragma HLS INTERFACE s_axilite port=B     bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;
    const int m = M;

    // Local buffers for tiling to enable pipelining and parallel access
    double localC[N][N];
    double localA[N][M];
    double localB[N][M];

#pragma HLS ARRAY_PARTITION variable=localC cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=localA cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=localB cyclic factor=4 dim=2

    // Load C from global memory
    load_C_outer: for (int i = 0; i < n; i++) {
        load_C_inner: for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            localC[i][j] = C[i][j];
        }
    }

    // Load A from global memory
    load_A_outer: for (int i = 0; i < n; i++) {
        load_A_inner: for (int k = 0; k < m; k++) {
#pragma HLS PIPELINE II=1
            localA[i][k] = A[i][k];
        }
    }

    // Load B from global memory
    load_B_outer: for (int i = 0; i < n; i++) {
        load_B_inner: for (int k = 0; k < m; k++) {
#pragma HLS PIPELINE II=1
            localB[i][k] = B[i][k];
        }
    }

    int i, j, k;

    // Beta scaling pass
    scale_i: for (i = 0; i < n; i++) {
        scale_j: for (j = 0; j <= i; j++) {
#pragma HLS PIPELINE II=1
            localC[i][j] *= beta;
        }
    }

    // Main compute: C[i][j] += A[j][k]*alpha*B[i][k] + B[j][k]*alpha*A[i][k]
    compute_i: for (i = 0; i < n; i++) {
        compute_k: for (k = 0; k < m; k++) {
            compute_j: for (j = 0; j <= i; j++) {
#pragma HLS PIPELINE II=1
                localC[i][j] += localA[j][k] * alpha * localB[i][k]
                              + localB[j][k] * alpha * localA[i][k];
            }
        }
    }

    // Store C back to global memory
    store_C_outer: for (int i = 0; i < n; i++) {
        store_C_inner: for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            C[i][j] = localC[i][j];
        }
    }
}

} // extern "C"