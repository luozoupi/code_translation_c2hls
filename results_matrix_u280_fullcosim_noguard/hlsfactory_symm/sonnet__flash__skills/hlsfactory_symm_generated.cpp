#include "symm.h"

extern "C" {

void kernel_symm(
        double alpha,
        double beta,
        double C[M][N],
        double A[M][M],
        double B[M][N])
{
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=alpha  bundle=control
#pragma HLS INTERFACE s_axilite port=beta   bundle=control
#pragma HLS INTERFACE s_axilite port=C      bundle=control
#pragma HLS INTERFACE s_axilite port=A      bundle=control
#pragma HLS INTERFACE s_axilite port=B      bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local copies for fast on-chip access
    double lC[M][N];
    double lA[M][M];
    double lB[M][N];

#pragma HLS ARRAY_PARTITION variable=lC cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=lA cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=lB cyclic factor=8 dim=2

    // Load C from global memory
    load_C:
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            lC[i][j] = C[i][j];
        }
    }

    // Load A from global memory
    load_A:
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < M; k++) {
#pragma HLS PIPELINE II=1
            lA[i][k] = A[i][k];
        }
    }

    // Load B from global memory
    load_B:
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            lB[i][j] = B[i][j];
        }
    }

    // Compute kernel
    const int m = M;
    const int n = N;

    int i, j, k;
    double temp2;

    compute_i:
    for (i = 0; i < m; i++) {
        compute_j:
        for (j = 0; j < n; j++) {
            temp2 = 0;
            compute_k:
            for (k = 0; k < i; k++) {
#pragma HLS PIPELINE II=1
                lC[k][j] += alpha * lB[i][j] * lA[i][k];
                temp2 += lB[k][j] * lA[i][k];
            }
            lC[i][j] = beta * lC[i][j] + alpha * lB[i][j] * lA[i][i] + alpha * temp2;
        }
    }

    // Store C back to global memory
    store_C:
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            C[i][j] = lC[i][j];
        }
    }
}

} // extern "C"