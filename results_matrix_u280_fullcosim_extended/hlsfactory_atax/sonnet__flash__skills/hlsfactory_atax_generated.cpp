#include "atax.h"

extern "C" {

void kernel_atax(
    double A[M + 0][N + 0],
    double x[N + 0],
    double y[N + 0],
    double tmp[M + 0])
{
#pragma HLS INTERFACE m_axi port=A   offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=x   offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=y   offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=tmp offset=slave bundle=gmem3
#pragma HLS INTERFACE s_axilite port=A      bundle=control
#pragma HLS INTERFACE s_axilite port=x      bundle=control
#pragma HLS INTERFACE s_axilite port=y      bundle=control
#pragma HLS INTERFACE s_axilite port=tmp    bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local buffers to avoid repeated global memory traffic
    double l_x[N];
    double l_y[N];
    double l_tmp[M];

#pragma HLS ARRAY_PARTITION variable=l_x   cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=l_y   cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=l_tmp cyclic factor=4 dim=1

    int i, j;

    // Load x from global memory
    load_x: for (j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
        l_x[j] = x[j];
    }

    // Initialize y
    init_y: for (i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        l_y[i] = 0.0;
    }

    // Main computation
    outer: for (i = 0; i < M; i++) {
        double tmp_i = 0.0;

        // Compute tmp[i] = A[i][:] * x[:]
        // Serial FP reduction — do NOT unroll to preserve bit-exact order
        inner1: for (j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            tmp_i = tmp_i + A[i][j] * l_x[j];
        }
        l_tmp[i] = tmp_i;

        // Accumulate y[j] += A[i][j] * tmp[i]
        // Serial FP reduction — do NOT unroll to preserve bit-exact order
        inner2: for (j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            l_y[j] = l_y[j] + A[i][j] * tmp_i;
        }
    }

    // Write tmp to global memory
    store_tmp: for (i = 0; i < M; i++) {
#pragma HLS PIPELINE II=1
        tmp[i] = l_tmp[i];
    }

    // Write y to global memory
    store_y: for (j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
        y[j] = l_y[j];
    }
}

} // extern "C"