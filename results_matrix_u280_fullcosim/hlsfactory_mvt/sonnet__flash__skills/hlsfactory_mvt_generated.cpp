#include "mvt.h"

extern "C" {

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
#pragma HLS INTERFACE s_axilite port=x1     bundle=control
#pragma HLS INTERFACE s_axilite port=x2     bundle=control
#pragma HLS INTERFACE s_axilite port=y_1    bundle=control
#pragma HLS INTERFACE s_axilite port=y_2    bundle=control
#pragma HLS INTERFACE s_axilite port=A      bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local caches for y vectors and x accumulators
    double local_y1[N];
    double local_y2[N];
    double local_x1[N];
    double local_x2[N];
    double local_A[N][N];

#pragma HLS ARRAY_PARTITION variable=local_y1 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=local_y2 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=local_x1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=local_x2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=local_A  cyclic factor=8 dim=2

    const int n = N;
    int i, j;

    // Load y_1 into local buffer
    load_y1: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        local_y1[i] = y_1[i];
    }

    // Load y_2 into local buffer
    load_y2: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        local_y2[i] = y_2[i];
    }

    // Load x1 into local buffer
    load_x1: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        local_x1[i] = x1[i];
    }

    // Load x2 into local buffer
    load_x2: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        local_x2[i] = x2[i];
    }

    // Load A into local buffer row by row
    load_A: for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            local_A[i][j] = A[i][j];
        }
    }

    // Loop 1: x1[i] += sum_j A[i][j] * y_1[j]
    loop1_i: for (i = 0; i < n; i++) {
        double acc = 0.0;
        loop1_j: for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
            acc += local_A[i][j] * local_y1[j];
        }
        local_x1[i] += acc;
    }

    // Loop 2: x2[i] += sum_j A[j][i] * y_2[j]
    loop2_i: for (i = 0; i < n; i++) {
        double acc = 0.0;
        loop2_j: for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
            acc += local_A[j][i] * local_y2[j];
        }
        local_x2[i] += acc;
    }

    // Write back x1
    store_x1: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        x1[i] = local_x1[i];
    }

    // Write back x2
    store_x2: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        x2[i] = local_x2[i];
    }
}

} // extern "C"