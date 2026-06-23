#include "mvt.h"

void kernel_mvt(
		double x1[ N + 0],
		double x2[ N + 0],
		double y_1[ N + 0],
		double y_2[ N + 0],
		double A[ N + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=x1  offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=x2  offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=y_1 offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=y_2 offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=A   offset=slave bundle=gmem4
#pragma HLS INTERFACE s_axilite port=x1  bundle=control
#pragma HLS INTERFACE s_axilite port=x2  bundle=control
#pragma HLS INTERFACE s_axilite port=y_1 bundle=control
#pragma HLS INTERFACE s_axilite port=y_2 bundle=control
#pragma HLS INTERFACE s_axilite port=A   bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;
    int i, j;

    // Local staging buffers for reuse across iterations
    double l_x1[N];
    double l_x2[N];
    double l_y1[N];
    double l_y2[N];
#pragma HLS ARRAY_PARTITION variable=l_y1 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_y2 cyclic factor=8 dim=1

    // Stage inputs into local memory
    load_loop: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        l_x1[i] = x1[i];
        l_x2[i] = x2[i];
        l_y1[i] = y_1[i];
        l_y2[i] = y_2[i];
    }

    // First MV: x1[i] += A[i][j] * y_1[j]  (row-major friendly)
    mvt1_i: for (i = 0; i < n; i++) {
        double acc = l_x1[i];
        mvt1_j: for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
            acc += A[i][j] * l_y1[j];
        }
        l_x1[i] = acc;
    }

    // Second MV: x2[i] += A[j][i] * y_2[j]
    mvt2_i: for (i = 0; i < n; i++) {
        double acc = l_x2[i];
        mvt2_j: for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            acc += A[j][i] * l_y2[j];
        }
        l_x2[i] = acc;
    }

    // Write back results
    store_loop: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        x1[i] = l_x1[i];
        x2[i] = l_x2[i];
    }
}