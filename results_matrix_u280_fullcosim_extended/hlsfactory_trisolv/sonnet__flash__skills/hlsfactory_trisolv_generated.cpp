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

    // Local buffers to avoid repeated global memory accesses
    double L_local[N][N];
    double x_local[N];
    double b_local[N];

    #pragma HLS ARRAY_PARTITION variable=L_local cyclic factor=4 dim=2
    #pragma HLS ARRAY_PARTITION variable=x_local cyclic factor=4 dim=1
    #pragma HLS ARRAY_PARTITION variable=b_local complete dim=1

    // Load b into local buffer
    load_b: for (int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        b_local[i] = b[i];
    }

    // Load L into local buffer
    load_L: for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            L_local[i][j] = L[i][j];
        }
    }

    const int n = N;
    int i, j;

    // Main triangular solve
    // Outer loop has loop-carried dependency on x_local, cannot pipeline
    for (i = 0; i < n; i++) {
        double xi = b_local[i];
        // Inner FP reduction: preserve serial order for bit-exact correctness
        for (j = 0; j < i; j++) {
            #pragma HLS PIPELINE II=1
            xi -= L_local[i][j] * x_local[j];
        }
        x_local[i] = xi / L_local[i][i];
    }

    // Write x back to global memory
    store_x: for (int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        x[i] = x_local[i];
    }
}

} // extern "C"