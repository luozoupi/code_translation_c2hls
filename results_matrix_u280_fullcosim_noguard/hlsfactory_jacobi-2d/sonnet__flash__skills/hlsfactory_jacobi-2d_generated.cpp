#include "jacobi-2d.h"

extern "C" {

void kernel_jacobi_2d(
                double A[N + 0][N + 0],
                double B[N + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;
    const int tsteps = TSTEPS;

    // Local ping-pong buffers to avoid repeated global memory access
    static double lA[N][N];
    static double lB[N][N];

#pragma HLS ARRAY_PARTITION variable=lA cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=lB cyclic factor=4 dim=2

    // Load A into local buffer
    load_A:
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            lA[i][j] = A[i][j];
        }
    }

    int t, i, j;

    for (t = 0; t < tsteps; t++) {

        // B[i][j] = 0.2 * (A[i][j] + A[i][j-1] + A[i][1+j] + A[1+i][j] + A[i-1][j])
        update_B:
        for (i = 1; i < n - 1; i++) {
            for (j = 1; j < n - 1; j++) {
#pragma HLS PIPELINE II=1
                lB[i][j] = 0.2 * (lA[i][j] + lA[i][j-1] + lA[i][1+j] + lA[1+i][j] + lA[i-1][j]);
            }
        }

        // A[i][j] = 0.2 * (B[i][j] + B[i][j-1] + B[i][1+j] + B[1+i][j] + B[i-1][j])
        update_A:
        for (i = 1; i < n - 1; i++) {
            for (j = 1; j < n - 1; j++) {
#pragma HLS PIPELINE II=1
                lA[i][j] = 0.2 * (lB[i][j] + lB[i][j-1] + lB[i][1+j] + lB[1+i][j] + lB[i-1][j]);
            }
        }
    }

    // Write back lA to global A
    store_A:
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            A[i][j] = lA[i][j];
        }
    }

    // Write back lB to global B
    store_B:
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            B[i][j] = lB[i][j];
        }
    }
}

} // extern "C"