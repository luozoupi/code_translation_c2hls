#include "atax.h"

extern "C" {

void kernel_atax(
        double A[M + 0][N + 0],
        double x[N + 0],
        double y[N + 0],
        double tmp[M + 0])
{
    #pragma HLS INTERFACE m_axi port=A   offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=x   offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=y   offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=tmp offset=slave bundle=gmem
    #pragma HLS INTERFACE s_axilite port=A      bundle=control
    #pragma HLS INTERFACE s_axilite port=x      bundle=control
    #pragma HLS INTERFACE s_axilite port=y      bundle=control
    #pragma HLS INTERFACE s_axilite port=tmp    bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local buffers to avoid repeated global memory traffic
    double l_A[M][N];
    double l_x[N];
    double l_y[N];
    double l_tmp[M];

    #pragma HLS ARRAY_PARTITION variable=l_A   cyclic factor=8 dim=2
    #pragma HLS ARRAY_PARTITION variable=l_x   cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=l_y   cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=l_tmp complete dim=1

    // Load x from global memory
    load_x: for (int j = 0; j < N; j++) {
        #pragma HLS PIPELINE II=1
        l_x[j] = x[j];
    }

    // Load A from global memory
    load_A: for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            l_A[i][j] = A[i][j];
        }
    }

    // Initialize y
    init_y: for (int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        l_y[i] = 0.0;
    }

    // Main computation
    outer: for (int i = 0; i < M; i++) {
        double tmp_i = 0.0;

        // Compute tmp[i] = sum_j A[i][j] * x[j]
        inner1: for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS UNROLL factor=8
            tmp_i += l_A[i][j] * l_x[j];
        }
        l_tmp[i] = tmp_i;

        // Accumulate y[j] += A[i][j] * tmp[i]
        inner2: for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS UNROLL factor=8
            l_y[j] += l_A[i][j] * tmp_i;
        }
    }

    // Write tmp back to global memory
    store_tmp: for (int i = 0; i < M; i++) {
        #pragma HLS PIPELINE II=1
        tmp[i] = l_tmp[i];
    }

    // Write y back to global memory
    store_y: for (int j = 0; j < N; j++) {
        #pragma HLS PIPELINE II=1
        y[j] = l_y[j];
    }
}

} // extern "C"