#include "atax.h"

extern "C" {

void kernel_atax( 
		 double A[ M + 0][N + 0],
		 double x[ N + 0],
		 double y[ N + 0],
		 double tmp[ M + 0])
{
#pragma HLS INTERFACE m_axi port=A   offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=x   offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=y   offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=tmp offset=slave bundle=gmem3
#pragma HLS INTERFACE s_axilite port=A   bundle=control
#pragma HLS INTERFACE s_axilite port=x   bundle=control
#pragma HLS INTERFACE s_axilite port=y   bundle=control
#pragma HLS INTERFACE s_axilite port=tmp bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local buffers to stage global memory reads for reuse
    double local_x[N];
    double local_y[N];
    double local_tmp[M];
    double local_A[M][N];

#pragma HLS ARRAY_PARTITION variable=local_x   cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=local_y   cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=local_tmp cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=local_A   cyclic factor=8 dim=2

    const int m = M;
    const int n = N;

    // Load x from global memory
    load_x: for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
        local_x[j] = x[j];
    }

    // Load A from global memory
    load_A: for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            local_A[i][j] = A[i][j];
        }
    }

    // Initialize y to 0
    init_y: for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        local_y[i] = 0.0;
    }

    // Main computation
    for (int i = 0; i < m; i++) {
        local_tmp[i] = 0.0;

        // Compute tmp[i] = sum_j A[i][j] * x[j]
        dot1: for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            local_tmp[i] = local_tmp[i] + local_A[i][j] * local_x[j];
        }

        // Accumulate y[j] += A[i][j] * tmp[i]
        dot2: for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            local_y[j] = local_y[j] + local_A[i][j] * local_tmp[i];
        }
    }

    // Write results back to global memory
    store_y: for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
        y[j] = local_y[j];
    }

    store_tmp: for (int i = 0; i < m; i++) {
#pragma HLS PIPELINE II=1
        tmp[i] = local_tmp[i];
    }
}

} // extern "C"