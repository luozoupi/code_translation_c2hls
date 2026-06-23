#include "mvt.h"

extern "C" {

void kernel_mvt(
        double x1[ N + 0],
        double x2[ N + 0],
        double y_1[ N + 0],
        double y_2[ N + 0],
        double A[ N + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=x1  offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=x2  offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=y_1 offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=y_2 offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=A   offset=slave bundle=gmem

#pragma HLS INTERFACE s_axilite port=x1      bundle=control
#pragma HLS INTERFACE s_axilite port=x2      bundle=control
#pragma HLS INTERFACE s_axilite port=y_1     bundle=control
#pragma HLS INTERFACE s_axilite port=y_2     bundle=control
#pragma HLS INTERFACE s_axilite port=A       bundle=control
#pragma HLS INTERFACE s_axilite port=return  bundle=control

    const int n = N;

    // Local buffers for y_1, y_2, and rows of A to enable pipelining
    double local_y1[N];
    double local_y2[N];
    double local_A[N];

#pragma HLS ARRAY_PARTITION variable=local_y1 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=local_y2 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=local_A  cyclic factor=8 dim=1

    // Load y_1 into local buffer
    load_y1: for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
        local_y1[j] = y_1[j];
    }

    // Load y_2 into local buffer
    load_y2: for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
        local_y2[j] = y_2[j];
    }

    int i, j;

    // First loop: x1[i] += A[i][j] * y_1[j]
    loop1_outer: for (i = 0; i < n; i++) {
        // Load row i of A into local buffer
        load_A_row: for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            local_A[j] = A[i][j];
        }

        double acc1 = x1[i];
        loop1_inner: for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
            acc1 += local_A[j] * local_y1[j];
        }
        x1[i] = acc1;
    }

    // Second loop: x2[i] += A[j][i] * y_2[j]
    loop2_outer: for (i = 0; i < n; i++) {
        // Load column i of A into local buffer (A[j][i] for j=0..n-1)
        load_A_col: for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            local_A[j] = A[j][i];
        }

        double acc2 = x2[i];
        loop2_inner: for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
            acc2 += local_A[j] * local_y2[j];
        }
        x2[i] = acc2;
    }

}

} // extern "C"