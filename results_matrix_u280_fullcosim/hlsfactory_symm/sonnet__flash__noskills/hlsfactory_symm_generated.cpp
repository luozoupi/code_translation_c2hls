#include "symm.h"

extern "C" {

void kernel_symm(
        double alpha,
        double beta,
        double C[M + 0][N + 0],
        double A[M + 0][M + 0],
        double B[M + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=alpha bundle=control
#pragma HLS INTERFACE s_axilite port=beta bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local buffers for parallel access
    double C_local[M][N];
    double A_local[M][M];
    double B_local[M][N];

#pragma HLS ARRAY_PARTITION variable=C_local cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=A_local cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=B_local cyclic factor=4 dim=2

    // Load A into local buffer
    load_A_outer: for (int i = 0; i < M; i++) {
        load_A_inner: for (int k = 0; k < M; k++) {
#pragma HLS PIPELINE II=1
            A_local[i][k] = A[i][k];
        }
    }

    // Load B into local buffer
    load_B_outer: for (int i = 0; i < M; i++) {
        load_B_inner: for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            B_local[i][j] = B[i][j];
        }
    }

    // Load C into local buffer
    load_C_outer: for (int i = 0; i < M; i++) {
        load_C_inner: for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            C_local[i][j] = C[i][j];
        }
    }

    // Compute — preserve original loop order exactly for correctness
    // The i-j-k structure has C[k][j] updated in the k loop (k < i),
    // so we must not pipeline across i iterations that alias on k.
    const int m = M;
    const int n = N;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double temp2 = 0.0;
            for (int k = 0; k < i; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=C_local inter false
                C_local[k][j] += alpha * B_local[i][j] * A_local[i][k];
                temp2 += B_local[k][j] * A_local[i][k];
            }
            C_local[i][j] = beta * C_local[i][j] + alpha * B_local[i][j] * A_local[i][i] + alpha * temp2;
        }
    }

    // Store C back
    store_C_outer: for (int i = 0; i < M; i++) {
        store_C_inner: for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            C[i][j] = C_local[i][j];
        }
    }
}

} // extern "C"