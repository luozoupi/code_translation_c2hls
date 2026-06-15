#include "trisolv.h"

extern "C" {

void kernel_trisolv(
        double L[N + 0][N + 0],
        double x[N + 0],
        double b[N + 0])
{
#pragma HLS INTERFACE m_axi port=L offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=L bundle=control
#pragma HLS INTERFACE s_axilite port=x bundle=control
#pragma HLS INTERFACE s_axilite port=b bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;

    // Local buffers to avoid repeated global-memory accesses
    double x_local[N];
    double b_local[N];
    double L_row[N];

#pragma HLS ARRAY_PARTITION variable=x_local cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=b_local cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=L_row  cyclic factor=8 dim=1

    // Load b into local buffer
    load_b: for (int k = 0; k < n; k++) {
#pragma HLS PIPELINE II=1
        b_local[k] = b[k];
    }

    // Initialize x_local from b_local
    init_x: for (int k = 0; k < n; k++) {
#pragma HLS PIPELINE II=1
        x_local[k] = b_local[k];
    }

    // Main triangular solve loop
    // Outer loop has loop-carried dependency: x[i] depends on x[j<i]
    // Cannot pipeline outer loop; pipeline inner loop instead
    outer: for (int i = 0; i < n; i++) {

        // Load row i of L into local buffer
        load_row: for (int k = 0; k <= i; k++) {
#pragma HLS PIPELINE II=1
            L_row[k] = L[i][k];
        }

        // Inner loop: accumulate dot product L[i][0..i-1] * x[0..i-1]
        // This loop is free of loop-carried dependencies within the same i
        double acc = 0.0;
        inner: for (int j = 0; j < i; j++) {
#pragma HLS PIPELINE II=1
            acc += L_row[j] * x_local[j];
        }

        // Update x[i]
        x_local[i] = (x_local[i] - acc) / L_row[i];
    }

    // Write x_local back to global memory
    store_x: for (int k = 0; k < n; k++) {
#pragma HLS PIPELINE II=1
        x[k] = x_local[k];
    }
}

} // extern "C"