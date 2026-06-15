#include "syr2k.h"

extern "C" {

void kernel_syr2k(
		  double alpha,
		  double beta,
		  double C[ N + 0][N + 0],
		  double A[ N + 0][M + 0],
		  double B[ N + 0][M + 0])
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
    double local_C[N][N];
    double local_A[N][M];
    double local_B[N][M];

#pragma HLS ARRAY_PARTITION variable=local_C cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=local_A cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=local_B cyclic factor=8 dim=2

    // Load C
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            local_C[i][j] = C[i][j];
        }
    }

    // Load A
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
            local_A[i][j] = A[i][j];
        }
    }

    // Load B
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
            local_B[i][j] = B[i][j];
        }
    }

    const int n = N;
    const int m = M;

    int i, j, k;
    for (i = 0; i < n; i++) {
        for (j = 0; j <= i; j++) {
#pragma HLS PIPELINE II=1
            local_C[i][j] *= beta;
        }
        for (k = 0; k < m; k++) {
            for (j = 0; j <= i; j++) {
#pragma HLS PIPELINE II=1
                local_C[i][j] += local_A[j][k]*alpha*local_B[i][k] + local_B[j][k]*alpha*local_A[i][k];
            }
        }
    }

    // Store C
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            C[i][j] = local_C[i][j];
        }
    }
}

} // extern "C"