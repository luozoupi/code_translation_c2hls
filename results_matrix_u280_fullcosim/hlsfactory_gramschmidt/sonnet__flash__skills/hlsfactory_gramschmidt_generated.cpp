#include "gramschmidt.h"

extern "C" {

void kernel_gramschmidt(
    double A[M + 0][N + 0],
    double R[N + 0][N + 0],
    double Q[M + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=R offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=Q offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=R bundle=control
#pragma HLS INTERFACE s_axilite port=Q bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local working copies for efficient BRAM access
    double localA[M][N];
    double localR[N][N];
    double localQ[M][N];

#pragma HLS ARRAY_PARTITION variable=localA cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=localQ cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=localR cyclic factor=4 dim=1

    // Load A into local memory
    load_A_outer: for (int i = 0; i < M; i++) {
        load_A_inner: for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            localA[i][j] = A[i][j];
        }
    }

    const int m = M;
    const int n = N;

    int i, j, k;
    double nrm;

    for (k = 0; k < n; k++) {
        nrm = 0.0;

        nrm_loop: for (i = 0; i < m; i++) {
#pragma HLS PIPELINE II=1
            nrm += localA[i][k] * localA[i][k];
        }

        localR[k][k] = sqrt(nrm);

        norm_q_loop: for (i = 0; i < m; i++) {
#pragma HLS PIPELINE II=1
            localQ[i][k] = localA[i][k] / localR[k][k];
        }

        j_loop: for (j = k + 1; j < n; j++) {
            localR[k][j] = 0.0;

            rkj_loop: for (i = 0; i < m; i++) {
#pragma HLS PIPELINE II=1
                localR[k][j] += localQ[i][k] * localA[i][j];
            }

            update_a_loop: for (i = 0; i < m; i++) {
#pragma HLS PIPELINE II=1
                localA[i][j] = localA[i][j] - localQ[i][k] * localR[k][j];
            }
        }
    }

    // Store results back to global memory
    store_R_outer: for (int i = 0; i < N; i++) {
        store_R_inner: for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            R[i][j] = localR[i][j];
        }
    }

    store_Q_outer: for (int i = 0; i < M; i++) {
        store_Q_inner: for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            Q[i][j] = localQ[i][j];
        }
    }

    store_A_outer: for (int i = 0; i < M; i++) {
        store_A_inner: for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            A[i][j] = localA[i][j];
        }
    }
}

} // extern "C"