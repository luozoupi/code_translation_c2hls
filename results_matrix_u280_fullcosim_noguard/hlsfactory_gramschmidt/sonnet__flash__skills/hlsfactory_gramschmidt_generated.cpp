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

    // Local arrays to avoid repeated global memory accesses
    double A_local[M][N];
    double R_local[N][N];
    double Q_local[M][N];

#pragma HLS ARRAY_PARTITION variable=A_local cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=Q_local cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=R_local cyclic factor=4 dim=2

    // Load A from global memory
    load_A_row:
    for (int i = 0; i < M; i++) {
#pragma HLS PIPELINE II=1
        load_A_col:
        for (int j = 0; j < N; j++) {
            A_local[i][j] = A[i][j];
        }
    }

    const int m = M;
    const int n = N;

    int i, j, k;
    double nrm;

    for (k = 0; k < n; k++) {
        nrm = 0.0;

        // Compute norm
        nrm_loop:
        for (i = 0; i < m; i++) {
#pragma HLS PIPELINE II=1
            nrm += A_local[i][k] * A_local[i][k];
        }

        R_local[k][k] = sqrt(nrm);

        // Compute Q column k
        q_loop:
        for (i = 0; i < m; i++) {
#pragma HLS PIPELINE II=1
            Q_local[i][k] = A_local[i][k] / R_local[k][k];
        }

        // Update R and A for remaining columns
        j_loop:
        for (j = k + 1; j < n; j++) {
            R_local[k][j] = 0.0;

            // Compute R[k][j]
            r_loop:
            for (i = 0; i < m; i++) {
#pragma HLS PIPELINE II=1
                R_local[k][j] += Q_local[i][k] * A_local[i][j];
            }

            // Update A column j
            a_update_loop:
            for (i = 0; i < m; i++) {
#pragma HLS PIPELINE II=1
                A_local[i][j] = A_local[i][j] - Q_local[i][k] * R_local[k][j];
            }
        }
    }

    // Write Q to global memory
    store_Q_row:
    for (int i = 0; i < M; i++) {
#pragma HLS PIPELINE II=1
        store_Q_col:
        for (int j = 0; j < N; j++) {
            Q[i][j] = Q_local[i][j];
        }
    }

    // Write R to global memory
    store_R_row:
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        store_R_col:
        for (int j = 0; j < N; j++) {
            R[i][j] = R_local[i][j];
        }
    }

    // Write A back to global memory (modified in-place)
    store_A_row:
    for (int i = 0; i < M; i++) {
#pragma HLS PIPELINE II=1
        store_A_col:
        for (int j = 0; j < N; j++) {
            A[i][j] = A_local[i][j];
        }
    }
}

} // extern "C"