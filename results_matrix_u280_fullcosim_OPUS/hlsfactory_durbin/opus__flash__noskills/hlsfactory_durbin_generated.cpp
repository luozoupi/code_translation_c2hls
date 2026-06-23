#include "durbin.h"

extern "C" {
void kernel_durbin(
		   double r[ N + 0],
		   double y[ N + 0])
{
#pragma HLS INTERFACE m_axi port=r offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=y offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=r bundle=control
#pragma HLS INTERFACE s_axilite port=y bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;

    double z[N];
    double y_local[N];
    double r_local[N];
#pragma HLS ARRAY_PARTITION variable=z cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=y_local cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=r_local cyclic factor=8 dim=1

    double alpha;
    double beta;
    double sum;

    int i, k;

    // Load r into local buffer
    load_r:
    for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        r_local[i] = r[i];
    }

    y_local[0] = -r_local[0];
    beta = 1.0;
    alpha = -r_local[0];

    main_loop:
    for (k = 1; k < n; k++) {
        beta = (1 - alpha * alpha) * beta;
        sum = 0.0;
        sum_loop:
        for (i = 0; i < k; i++) {
#pragma HLS PIPELINE II=1
            sum += r_local[k - i - 1] * y_local[i];
        }
        alpha = -(r_local[k] + sum) / beta;

        update_z:
        for (i = 0; i < k; i++) {
#pragma HLS PIPELINE II=1
            z[i] = y_local[i] + alpha * y_local[k - i - 1];
        }
        copy_y:
        for (i = 0; i < k; i++) {
#pragma HLS PIPELINE II=1
            y_local[i] = z[i];
        }
        y_local[k] = alpha;
    }

    // Write back y
    store_y:
    for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        y[i] = y_local[i];
    }
}
}