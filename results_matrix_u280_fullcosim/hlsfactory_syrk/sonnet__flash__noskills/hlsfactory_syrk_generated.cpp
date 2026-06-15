#include "syrk.h"

extern "C" {

void kernel_syrk(
        double alpha,
        double beta,
        double C[N + 0][N + 0],
        double A[N + 0][M + 0])
{
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=alpha bundle=control
#pragma HLS INTERFACE s_axilite port=beta bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local copies for efficient on-chip access
    double C_local[N][N];
    double A_local[N][M];

#pragma HLS ARRAY_PARTITION variable=C_local cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=A_local cyclic factor=8 dim=2

    // Load C into local buffer
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            C_local[i][j] = C[i][j];
        }
    }

    // Load A into local buffer
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
            A_local[i][j] = A[i][j];
        }
    }

    const int n = N;
    const int m = M;

    int i, j, k;

    for (i = 0; i < n; i++) {
        // Scale C[i][0..i] by beta
        for (j = 0; j <= i; j++) {
#pragma HLS PIPELINE II=1
            C_local[i][j] *= beta;
        }
        // Accumulate alpha * A[i][k] * A[j][k]
        for (k = 0; k < m; k++) {
            for (j = 0; j <= i; j++) {
#pragma HLS PIPELINE II=1
                C_local[i][j] += alpha * A_local[i][k] * A_local[j][k];
            }
        }
    }

    // Write results back
    for (int ii = 0; ii < N; ii++) {
        for (int jj = 0; jj < N; jj++) {
#pragma HLS PIPELINE II=1
            C[ii][jj] = C_local[ii][jj];
        }
    }
}

} // extern "C"