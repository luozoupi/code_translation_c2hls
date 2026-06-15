#include "gesummv.h"

extern "C" {

void kernel_gesummv(
        double alpha,
        double beta,
        double A[ N + 0][N + 0],
        double B[ N + 0][N + 0],
        double tmp[ N + 0],
        double x[ N + 0],
        double y[ N + 0])
{
#pragma HLS INTERFACE m_axi port=A   offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=B   offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=tmp offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=x   offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=y   offset=slave bundle=gmem4

#pragma HLS INTERFACE s_axilite port=alpha  bundle=control
#pragma HLS INTERFACE s_axilite port=beta   bundle=control
#pragma HLS INTERFACE s_axilite port=A      bundle=control
#pragma HLS INTERFACE s_axilite port=B      bundle=control
#pragma HLS INTERFACE s_axilite port=tmp    bundle=control
#pragma HLS INTERFACE s_axilite port=x      bundle=control
#pragma HLS INTERFACE s_axilite port=y      bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local copies for better memory access patterns
    double x_local[N];
    double tmp_local[N];
    double y_local[N];

#pragma HLS ARRAY_PARTITION variable=x_local   cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=tmp_local cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=y_local   cyclic factor=4 dim=1

    // Load x into local buffer
    load_x: for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        x_local[i] = x[i];
    }

    const int n = N;
    int i, j;

    outer_loop: for (i = 0; i < n; i++) {
        double tmp_i = 0.0;
        double y_i   = 0.0;

        inner_loop: for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            tmp_i += A[i][j] * x_local[j];
            y_i   += B[i][j] * x_local[j];
        }

        tmp_local[i] = tmp_i;
        y_local[i]   = alpha * tmp_i + beta * y_i;
    }

    // Write results back
    store_tmp: for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        tmp[i] = tmp_local[i];
    }

    store_y: for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        y[i] = y_local[i];
    }
}

} // extern "C"