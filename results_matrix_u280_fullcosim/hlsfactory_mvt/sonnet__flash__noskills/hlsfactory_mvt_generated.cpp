#include "mvt.h"

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
#pragma HLS INTERFACE s_axilite port=x1     bundle=control
#pragma HLS INTERFACE s_axilite port=x2     bundle=control
#pragma HLS INTERFACE s_axilite port=y_1    bundle=control
#pragma HLS INTERFACE s_axilite port=y_2    bundle=control
#pragma HLS INTERFACE s_axilite port=A      bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local buffers for reduced memory traffic
    double local_x1[N];
    double local_x2[N];
    double local_y1[N];
    double local_y2[N];
    double local_A[N][N];

#pragma HLS ARRAY_PARTITION variable=local_x1 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=local_x2 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=local_y1 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=local_y2 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=local_A  cyclic factor=8 dim=2

    // Load inputs into local buffers
    load_x1: for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        local_x1[i] = x1[i];
    }
    load_x2: for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        local_x2[i] = x2[i];
    }
    load_y1: for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
        local_y1[j] = y_1[j];
    }
    load_y2: for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
        local_y2[j] = y_2[j];
    }
    load_A: for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            local_A[i][j] = A[i][j];
        }
    }

    const int n = N;
    int i, j;

    // First loop: x1[i] += A[i][j] * y_1[j]
    for (i = 0; i < n; i++) {
        double sum1 = local_x1[i];
        for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            sum1 += local_A[i][j] * local_y1[j];
        }
        local_x1[i] = sum1;
    }

    // Second loop: x2[i] += A[j][i] * y_2[j]
    for (i = 0; i < n; i++) {
        double sum2 = local_x2[i];
        for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            sum2 += local_A[j][i] * local_y2[j];
        }
        local_x2[i] = sum2;
    }

    // Store results back
    store_x1: for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        x1[i] = local_x1[i];
    }
    store_x2: for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        x2[i] = local_x2[i];
    }
}