#include "atax.h"

extern "C" {

void kernel_atax(
        double A[ M + 0][N + 0],
        double x[ N + 0],
        double y[ N + 0],
        double tmp[ M + 0])
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

    // Local copies for fast access
    double local_A[M][N];
    double local_x[N];
    double local_y[N];
    double local_tmp[M];

#pragma HLS ARRAY_PARTITION variable=local_A cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=local_x cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=local_y cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=local_tmp complete

    // Load x
    for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
        local_x[j] = x[j];
    }

    // Load A
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            local_A[i][j] = A[i][j];
        }
    }

    // Initialize y
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        local_y[i] = 0.0;
    }

    // Compute tmp and accumulate into y
    for (int i = 0; i < M; i++) {
        double tmp_i = 0.0;

        for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
            tmp_i += local_A[i][j] * local_x[j];
        }

        local_tmp[i] = tmp_i;

        for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
            local_y[j] += local_A[i][j] * tmp_i;
        }
    }

    // Write back tmp
    for (int i = 0; i < M; i++) {
#pragma HLS PIPELINE II=1
        tmp[i] = local_tmp[i];
    }

    // Write back y
    for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
        y[j] = local_y[j];
    }
}

} // extern "C"