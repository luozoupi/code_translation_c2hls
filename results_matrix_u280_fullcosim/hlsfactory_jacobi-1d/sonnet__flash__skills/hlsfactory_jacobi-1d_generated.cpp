#include "jacobi-1d.h"

extern "C" {

void kernel_jacobi_1d(
    double A[N + 0],
    double B[N + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;
    const int tsteps = TSTEPS;

    // Local buffers to avoid repeated global memory traffic across timesteps
    double lA[N];
    double lB[N];
#pragma HLS ARRAY_PARTITION variable=lA cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=lB cyclic factor=4 dim=1

    // Load A from global memory into local buffer
    load_A: for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        lA[i] = A[i];
    }

    // Load B from global memory into local buffer
    load_B: for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        lB[i] = B[i];
    }

    // Main Jacobi iteration - true dependency on t, so cannot pipeline outer loop
    for (int t = 0; t < tsteps; t++) {
        // Update B from A
        update_B: for (int i = 1; i < n - 1; i++) {
#pragma HLS PIPELINE II=1
            lB[i] = 0.33333 * (lA[i-1] + lA[i] + lA[i + 1]);
        }
        // Update A from B
        update_A: for (int i = 1; i < n - 1; i++) {
#pragma HLS PIPELINE II=1
            lA[i] = 0.33333 * (lB[i-1] + lB[i] + lB[i + 1]);
        }
    }

    // Write results back to global memory
    store_A: for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        A[i] = lA[i];
    }

    store_B: for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        B[i] = lB[i];
    }
}

} // extern "C"