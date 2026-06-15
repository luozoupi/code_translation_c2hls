#include "jacobi-2d.h"

extern "C" {

void kernel_jacobi_2d(
                double A[N + 0][N + 0],
                double B[N + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=8100
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=8100
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;
    const int tsteps = TSTEPS;

    // Local buffers staged in BRAM to avoid repeated AXI round-trips
    double lA[N][N];
    double lB[N][N];
    // Partition along columns (dim=2) for parallel j-neighbor access (j-1, j, j+1)
    // Partition along rows (dim=1) for parallel i-neighbor access (i-1, i, i+1)
#pragma HLS ARRAY_PARTITION variable=lA cyclic factor=2 dim=1
#pragma HLS ARRAY_PARTITION variable=lB cyclic factor=2 dim=1
#pragma HLS ARRAY_PARTITION variable=lA cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=lB cyclic factor=4 dim=2

    // Load A into local buffer (full array including boundary)
    load_A:
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            lA[i][j] = A[i][j];
            lB[i][j] = 0.0;
        }
    }

    // Run time steps entirely in local BRAM
    for (int t = 0; t < tsteps; t++) {
        // Compute B from A (interior only; boundaries stay 0)
        stencil_B:
        for (int i = 1; i < n - 1; i++) {
#pragma HLS PIPELINE II=4
            for (int j = 1; j < n - 1; j++) {
                lB[i][j] = 0.2 * (lA[i][j] + lA[i][j-1] + lA[i][1+j] + lA[1+i][j] + lA[i-1][j]);
            }
        }
        // Compute A from B (interior only; boundaries unchanged)
        stencil_A:
        for (int i = 1; i < n - 1; i++) {
#pragma HLS PIPELINE II=4
            for (int j = 1; j < n - 1; j++) {
                lA[i][j] = 0.2 * (lB[i][j] + lB[i][j-1] + lB[i][1+j] + lB[1+i][j] + lB[i-1][j]);
            }
        }
    }

    // Write back results to global memory
    store_out:
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            A[i][j] = lA[i][j];
            B[i][j] = lB[i][j];
        }
    }
}

} // extern "C"