#include "gramschmidt.h"

extern "C" {

void kernel_gramschmidt(
        double A[M + 0][N + 0],
        double R[N + 0][N + 0],
        double Q[M + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=R offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=Q offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=R bundle=control
#pragma HLS INTERFACE s_axilite port=Q bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local copies for better data reuse and partitioning
    double lA[M][N];
    double lR[N][N];
    double lQ[M][N];

#pragma HLS ARRAY_PARTITION variable=lA cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=lR cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=lQ cyclic factor=4 dim=2

    // Load A into local buffer
    for (int i = 0; i < M; i++) {
#pragma HLS PIPELINE II=1
        for (int j = 0; j < N; j++) {
            lA[i][j] = A[i][j];
        }
    }

    const int m = M;
    const int n = N;

    int i, j, k;
    double nrm;

    for (k = 0; k < n; k++) {
        // FP reduction: keep serial, pipeline the loop body only
        nrm = 0.0;
        for (i = 0; i < m; i++) {
#pragma HLS PIPELINE II=1
            nrm += lA[i][k] * lA[i][k];
        }

        lR[k][k] = sqrt(nrm);

        // Normalize column: independent iterations, safe to pipeline
        for (i = 0; i < m; i++) {
#pragma HLS PIPELINE II=1
            lQ[i][k] = lA[i][k] / lR[k][k];
        }

        for (j = k + 1; j < n; j++) {
            // FP reduction: keep serial, pipeline the loop body only
            lR[k][j] = 0.0;
            for (i = 0; i < m; i++) {
#pragma HLS PIPELINE II=1
                lR[k][j] += lQ[i][k] * lA[i][j];
            }

            // Update A column: independent iterations, safe to pipeline
            for (i = 0; i < m; i++) {
#pragma HLS PIPELINE II=1
                lA[i][j] = lA[i][j] - lQ[i][k] * lR[k][j];
            }
        }
    }

    // Write back results
    for (int i = 0; i < M; i++) {
#pragma HLS PIPELINE II=1
        for (int j = 0; j < N; j++) {
            A[i][j] = lA[i][j];
        }
    }

    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        for (int j = 0; j < N; j++) {
            R[i][j] = lR[i][j];
        }
    }

    for (int i = 0; i < M; i++) {
#pragma HLS PIPELINE II=1
        for (int j = 0; j < N; j++) {
            Q[i][j] = lQ[i][j];
        }
    }
}

} // extern "C"