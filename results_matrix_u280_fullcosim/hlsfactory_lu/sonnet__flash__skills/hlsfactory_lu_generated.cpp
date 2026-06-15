#include "lu.h"

extern "C" {
void kernel_lu(
	       double A[ N + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local copy for efficient on-chip computation
    double A_local[N][N];
#pragma HLS ARRAY_PARTITION variable=A_local cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=A_local cyclic factor=8 dim=1

    // Load A into local buffer
    load_i: for (int i = 0; i < N; i++) {
        load_j: for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            A_local[i][j] = A[i][j];
        }
    }

    const int n = N;
    int i, j, k;

    for (i = 0; i < n; i++) {
        // First nested loop: j < i
        for (j = 0; j < i; j++) {
            // Inner k loop for first j-loop
            for (k = 0; k < j; k++) {
#pragma HLS PIPELINE II=1
                A_local[i][j] -= A_local[i][k] * A_local[k][j];
            }
            A_local[i][j] /= A_local[j][j];
        }
        // Second nested loop: j >= i
        for (j = i; j < n; j++) {
            // Inner k loop for second j-loop
            for (k = 0; k < i; k++) {
#pragma HLS PIPELINE II=1
                A_local[i][j] -= A_local[i][k] * A_local[k][j];
            }
        }
    }

    // Store local buffer back to A
    store_i: for (int i = 0; i < N; i++) {
        store_j: for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            A[i][j] = A_local[i][j];
        }
    }
}
}