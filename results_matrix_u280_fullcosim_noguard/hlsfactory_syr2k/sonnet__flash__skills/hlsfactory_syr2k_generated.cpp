#include "syr2k.h"

extern "C" {

void kernel_syr2k(
          double alpha,
          double beta,
          double C[N][N],
          double A[N][M],
          double B[N][M])
{
    #pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=alpha bundle=control
    #pragma HLS INTERFACE s_axilite port=beta  bundle=control
    #pragma HLS INTERFACE s_axilite port=C     bundle=control
    #pragma HLS INTERFACE s_axilite port=A     bundle=control
    #pragma HLS INTERFACE s_axilite port=B     bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local buffers to stage data for reuse across the k-loop
    double localC[N][N];
    double localA[N][M];
    double localB[N][M];

    #pragma HLS ARRAY_PARTITION variable=localC cyclic factor=8 dim=2
    #pragma HLS ARRAY_PARTITION variable=localA cyclic factor=8 dim=2
    #pragma HLS ARRAY_PARTITION variable=localB cyclic factor=8 dim=2

    const int n = N;
    const int m = M;

    // Load C into local buffer
    load_C_i: for (int i = 0; i < n; i++) {
        load_C_j: for (int j = 0; j < n; j++) {
            #pragma HLS PIPELINE II=1
            localC[i][j] = C[i][j];
        }
    }

    // Load A into local buffer
    load_A_i: for (int i = 0; i < n; i++) {
        load_A_k: for (int k = 0; k < m; k++) {
            #pragma HLS PIPELINE II=1
            localA[i][k] = A[i][k];
        }
    }

    // Load B into local buffer
    load_B_i: for (int i = 0; i < n; i++) {
        load_B_k: for (int k = 0; k < m; k++) {
            #pragma HLS PIPELINE II=1
            localB[i][k] = B[i][k];
        }
    }

    // Compute: scale lower triangle of C by beta
    scale_i: for (int i = 0; i < n; i++) {
        scale_j: for (int j = 0; j <= i; j++) {
            #pragma HLS PIPELINE II=1
            localC[i][j] *= beta;
        }
    }

    // Compute: update lower triangle of C
    compute_i: for (int i = 0; i < n; i++) {
        compute_k: for (int k = 0; k < m; k++) {
            compute_j: for (int j = 0; j <= i; j++) {
                #pragma HLS PIPELINE II=1
                localC[i][j] += localA[j][k] * alpha * localB[i][k]
                              +  localB[j][k] * alpha * localA[i][k];
            }
        }
    }

    // Store result back to global memory (lower triangle only)
    store_i: for (int i = 0; i < n; i++) {
        store_j: for (int j = 0; j < n; j++) {
            #pragma HLS PIPELINE II=1
            C[i][j] = localC[i][j];
        }
    }
}

} // extern "C"