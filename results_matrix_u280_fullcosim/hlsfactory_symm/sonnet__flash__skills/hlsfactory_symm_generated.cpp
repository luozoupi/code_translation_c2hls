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

    const int m = M;
    const int n = N;

    // Local buffers for tiling/reuse
    double C_local[M][N];
    double A_local[M][M];
    double B_local[M][N];

#pragma HLS ARRAY_PARTITION variable=C_local cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=B_local cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=A_local complete dim=2

    // Load C into local buffer
    load_C_i: for (int i = 0; i < m; i++) {
        load_C_j: for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            C_local[i][j] = C[i][j];
        }
    }

    // Load A into local buffer
    load_A_i: for (int i = 0; i < m; i++) {
        load_A_k: for (int k = 0; k < m; k++) {
#pragma HLS PIPELINE II=1
            A_local[i][k] = A[i][k];
        }
    }

    // Load B into local buffer
    load_B_i: for (int i = 0; i < m; i++) {
        load_B_j: for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            B_local[i][j] = B[i][j];
        }
    }

    int i, j, k;
    double temp2;

    outer_i: for (i = 0; i < m; i++) {
        middle_j: for (j = 0; j < n; j++) {
            temp2 = 0;
            inner_k: for (k = 0; k < i; k++) {
#pragma HLS PIPELINE II=1
                C_local[k][j] += alpha * B_local[i][j] * A_local[i][k];
                temp2 += B_local[k][j] * A_local[i][k];
            }
            C_local[i][j] = beta * C_local[i][j] + alpha * B_local[i][j] * A_local[i][i] + alpha * temp2;
        }
    }

    // Store C back to global memory
    store_C_i: for (int i = 0; i < m; i++) {
        store_C_j: for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            C[i][j] = C_local[i][j];
        }
    }
}

} // extern "C"