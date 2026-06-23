#include "gesummv.h"

extern "C" {

void kernel_gesummv(
        double alpha,
        double beta,
        double A[N + 0][N + 0],
        double B[N + 0][N + 0],
        double tmp[N + 0],
        double x[N + 0],
        double y[N + 0])
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

    const int n = N;

    // Local buffers to avoid repeated global memory reads
    double x_local[N];
#pragma HLS ARRAY_PARTITION variable=x_local cyclic factor=4 dim=1

    double A_row[N];
#pragma HLS ARRAY_PARTITION variable=A_row cyclic factor=4 dim=1

    double B_row[N];
#pragma HLS ARRAY_PARTITION variable=B_row cyclic factor=4 dim=1

    // Load x into local buffer once
    load_x: for (int k = 0; k < n; k++) {
#pragma HLS PIPELINE II=1
        x_local[k] = x[k];
    }

    int i, j;

    for (i = 0; i < n; i++) {
        // Load row i of A and B into local buffers
        load_A_row: for (int k = 0; k < n; k++) {
#pragma HLS PIPELINE II=1
            A_row[k] = A[i][k];
            B_row[k] = B[i][k];
        }

        double tmp_i = 0.0;
        double y_i   = 0.0;

        // Inner compute loop — pipeline with local arrays for conflict-free access
        compute: for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
            tmp_i += A_row[j] * x_local[j];
            y_i   += B_row[j] * x_local[j];
        }

        tmp[i] = tmp_i;
        y[i]   = alpha * tmp_i + beta * y_i;
    }
}

} // extern "C"