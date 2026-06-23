#include "symm.h"

void kernel_symm(
		 double alpha,
		 double beta,
		 double C[ M + 0][N + 0],
		 double A[ M + 0][M + 0],
		 double B[ M + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=alpha bundle=control
#pragma HLS INTERFACE s_axilite port=beta bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int m = M;
    const int n = N;

    int i, j, k;

    // Stage the working set into local buffers for reuse and parallel access.
    static double C_local[M][N];
    static double A_local[M][M];
    static double B_local[M][N];
#pragma HLS ARRAY_PARTITION variable=A_local cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=B_local cyclic factor=8 dim=1

    // Load inputs into local memory.
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            C_local[i][j] = C[i][j];
            B_local[i][j] = B[i][j];
        }
    }
    for (i = 0; i < M; i++) {
        for (j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
            A_local[i][j] = A[i][j];
        }
    }

    // Main computation. The j loop iterations are independent, so pipeline it.
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            double temp2 = 0;
            double aii = A_local[i][i];
            double bij = B_local[i][j];
            for (k = 0; k < M; k++) {
#pragma HLS UNROLL
                if (k < i) {
                    C_local[k][j] += alpha * bij * A_local[i][k];
                    temp2 += B_local[k][j] * A_local[i][k];
                }
            }
            C_local[i][j] = beta * C_local[i][j] + alpha * bij * aii + alpha * temp2;
        }
    }

    // Store result back to global memory.
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            C[i][j] = C_local[i][j];
        }
    }
}