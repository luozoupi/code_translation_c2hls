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

    // Local buffers to avoid repeated global memory accesses
    static double lA[N][N];
    static double lB[N][N];

#pragma HLS ARRAY_PARTITION variable=lA cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=lB cyclic factor=8 dim=2

    // Copy A from global memory to local buffer
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            lA[i][j] = A[i][j];
        }
    }

    // Copy B from global memory to local buffer
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            lB[i][j] = B[i][j];
        }
    }

    int t, i, j;

    for (t = 0; t < tsteps; t++) {
        for (i = 1; i < n - 1; i++) {
            for (j = 1; j < n - 1; j++) {
#pragma HLS PIPELINE II=1
                lB[i][j] = 0.2 * (lA[i][j] + lA[i][j-1] + lA[i][1+j] + lA[1+i][j] + lA[i-1][j]);
            }
        }
        for (i = 1; i < n - 1; i++) {
            for (j = 1; j < n - 1; j++) {
#pragma HLS PIPELINE II=1
                lA[i][j] = 0.2 * (lB[i][j] + lB[i][j-1] + lB[i][1+j] + lB[1+i][j] + lB[i-1][j]);
            }
        }
    }

    // Write back local buffers to global memory
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            A[i][j] = lA[i][j];
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            B[i][j] = lB[i][j];
        }
    }
}

} // extern "C"