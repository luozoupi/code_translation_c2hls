#include "syrk.h"

void kernel_syrk(
    double alpha,
    double beta,
    double C[N + 0][N + 0],
    double A[N + 0][M + 0])
{
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=alpha bundle=control
#pragma HLS INTERFACE s_axilite port=beta  bundle=control
#pragma HLS INTERFACE s_axilite port=C     bundle=control
#pragma HLS INTERFACE s_axilite port=A     bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local copies to enable partitioning and fast local access
    double localC[N][N];
    double localA[N][M];

#pragma HLS ARRAY_PARTITION variable=localC cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=localA cyclic factor=8 dim=2

    // Load A into local buffer
    load_A_outer: for (int i = 0; i < N; i++) {
        load_A_inner: for (int k = 0; k < M; k++) {
#pragma HLS PIPELINE II=1
            localA[i][k] = A[i][k];
        }
    }

    // Load lower triangle of C into local buffer
    load_C_outer: for (int i = 0; i < N; i++) {
        load_C_inner: for (int j = 0; j <= i; j++) {
#pragma HLS PIPELINE II=1
            localC[i][j] = C[i][j];
        }
    }

    const int n = N;
    const int m = M;

    int i, j, k;

    // Scale lower triangle of C by beta
    scale_i: for (i = 0; i < n; i++) {
        scale_j: for (j = 0; j <= i; j++) {
#pragma HLS PIPELINE II=1
            localC[i][j] *= beta;
        }
    }

    // Symmetric rank-k update: C += alpha * A * A^T (lower triangle)
    update_i: for (i = 0; i < n; i++) {
        update_k: for (k = 0; k < m; k++) {
#pragma HLS PIPELINE II=1
            double aik = localA[i][k];
            update_j: for (j = 0; j <= i; j++) {
#pragma HLS UNROLL factor=8
                localC[i][j] += alpha * aik * localA[j][k];
            }
        }
    }

    // Store lower triangle of C back to global memory
    store_C_outer: for (int i = 0; i < N; i++) {
        store_C_inner: for (int j = 0; j <= i; j++) {
#pragma HLS PIPELINE II=1
            C[i][j] = localC[i][j];
        }
    }
}