#include "gesummv.h"

void kernel_gesummv(
		    double alpha,
		    double beta,
		    double A[ N + 0][N + 0],
		    double B[ N + 0][N + 0],
		    double tmp[ N + 0],
		    double x[ N + 0],
		    double y[ N + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=tmp offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=y offset=slave bundle=gmem4
#pragma HLS INTERFACE s_axilite port=alpha bundle=control
#pragma HLS INTERFACE s_axilite port=beta bundle=control
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=tmp bundle=control
#pragma HLS INTERFACE s_axilite port=x bundle=control
#pragma HLS INTERFACE s_axilite port=y bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;

    int i, j;

    // Local buffer for x to enable parallel access in the inner loop
    double x_local[N];
#pragma HLS ARRAY_PARTITION variable=x_local cyclic factor=8 dim=1

    for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        x_local[i] = x[i];
    }

    for (i = 0; i < n; i++) {
        double tmp_acc = 0.0;
        double y_acc = 0.0;

    inner_loop:
        for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            tmp_acc = A[i][j] * x_local[j] + tmp_acc;
            y_acc = B[i][j] * x_local[j] + y_acc;
        }

        tmp[i] = tmp_acc;
        y[i] = alpha * tmp_acc + beta * y_acc;
    }
}