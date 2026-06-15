#include "jacobi-1d.h"

extern "C" {

void kernel_jacobi_1d(
                double A[N + 0],
                double B[N + 0])
{
    #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
    #pragma HLS INTERFACE s_axilite port=A bundle=control
    #pragma HLS INTERFACE s_axilite port=B bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    double localA[N];
    double localB[N];

    #pragma HLS ARRAY_PARTITION variable=localA cyclic factor=4 dim=1
    #pragma HLS ARRAY_PARTITION variable=localB cyclic factor=4 dim=1

    // Load A into local buffer
    for (int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        localA[i] = A[i];
    }

    // Load B into local buffer
    for (int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        localB[i] = B[i];
    }

    const int n = N;
    const int tsteps = TSTEPS;

    for (int t = 0; t < tsteps; t++) {
        for (int i = 1; i < n - 1; i++) {
            #pragma HLS PIPELINE II=1
            localB[i] = 0.33333 * (localA[i-1] + localA[i] + localA[i + 1]);
        }
        for (int i = 1; i < n - 1; i++) {
            #pragma HLS PIPELINE II=1
            localA[i] = 0.33333 * (localB[i-1] + localB[i] + localB[i + 1]);
        }
    }

    // Store localA back to A
    for (int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        A[i] = localA[i];
    }

    // Store localB back to B
    for (int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        B[i] = localB[i];
    }
}

} // extern "C"