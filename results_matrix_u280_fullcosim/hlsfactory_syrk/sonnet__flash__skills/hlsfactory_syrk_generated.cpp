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

    // Local copies for on-chip computation
    double C_local[N][N];
    double A_local[N][M];

    #pragma HLS ARRAY_PARTITION variable=C_local cyclic factor=8 dim=2
    #pragma HLS ARRAY_PARTITION variable=A_local cyclic factor=8 dim=2

    // Load C into local buffer
    load_C_i: for (int i = 0; i < N; i++) {
        load_C_j: for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            C_local[i][j] = C[i][j];
        }
    }

    // Load A into local buffer
    load_A_i: for (int i = 0; i < N; i++) {
        load_A_k: for (int k = 0; k < M; k++) {
            #pragma HLS PIPELINE II=1
            A_local[i][k] = A[i][k];
        }
    }

    const int n = N;
    const int m = M;

    int i, j, k;

    // Beta scaling pass
    scale_i: for (i = 0; i < n; i++) {
        scale_j: for (j = 0; j <= i; j++) {
            #pragma HLS PIPELINE II=1
            C_local[i][j] *= beta;
        }
    }

    // Alpha * A * A^T accumulation pass
    compute_i: for (i = 0; i < n; i++) {
        compute_k: for (k = 0; k < m; k++) {
            #pragma HLS PIPELINE II=1
            double a_ik = A_local[i][k];
            compute_j: for (j = 0; j <= i; j++) {
                #pragma HLS UNROLL factor=8
                C_local[i][j] += alpha * a_ik * A_local[j][k];
            }
        }
    }

    // Store C_local back to global memory
    store_C_i: for (int i = 0; i < N; i++) {
        store_C_j: for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            C[i][j] = C_local[i][j];
        }
    }
}

} // extern "C"