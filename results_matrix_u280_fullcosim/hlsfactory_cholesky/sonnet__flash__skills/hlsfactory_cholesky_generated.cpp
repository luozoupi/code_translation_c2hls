#include "cholesky.h"

extern "C" {

void kernel_cholesky(double A[N + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem depth=14400
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local buffer for fast on-chip access with partitioning
    double L[N][N];
#pragma HLS ARRAY_PARTITION variable=L cyclic factor=8 dim=2

    // Load A into local buffer
    load_i: for (int i = 0; i < N; i++) {
        load_j: for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            L[i][j] = A[i][j];
        }
    }

    const int n = N;

    int i, j, k;

    for (i = 0; i < n; i++) {

        for (j = 0; j < i; j++) {
            // Reduction over k: loop-carried dep on A[i][j], pipeline with II=1
            double sum_ij = L[i][j];
            for (k = 0; k < j; k++) {
#pragma HLS PIPELINE II=1
                sum_ij -= L[i][k] * L[j][k];
            }
            L[i][j] = sum_ij / L[j][j];
        }

        // Reduction for diagonal element
        double sum_ii = L[i][i];
        for (k = 0; k < i; k++) {
#pragma HLS PIPELINE II=1
            sum_ii -= L[i][k] * L[i][k];
        }
        L[i][i] = sqrt(sum_ii);
    }

    // Store result back to A
    store_i: for (int i = 0; i < N; i++) {
        store_j: for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            A[i][j] = L[i][j];
        }
    }
}

} // extern "C"