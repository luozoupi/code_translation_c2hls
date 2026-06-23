#include "durbin.h"

extern "C" {
void kernel_durbin(
		   double r[ N + 0],
		   double y[ N + 0])
{
#pragma HLS INTERFACE m_axi port=r offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=y offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=r bundle=control
#pragma HLS INTERFACE s_axilite port=y bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;

    double z[N];
    double r_local[N];
    double y_local[N];

#pragma HLS ARRAY_PARTITION variable=z complete dim=1
#pragma HLS ARRAY_PARTITION variable=r_local complete dim=1
#pragma HLS ARRAY_PARTITION variable=y_local complete dim=1

    double alpha;
    double beta;
    double sum;

    int i, k;

    // Load r into local buffer
    for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        r_local[i] = r[i];
    }

    y_local[0] = -r_local[0];
    beta = 1.0;
    alpha = -r_local[0];

    for (k = 1; k < n; k++) {
        beta = (1 - alpha * alpha) * beta;
        sum = 0.0;

        for (i = 0; i < k; i++) {
#pragma HLS PIPELINE II=1
            sum += r_local[k - i - 1] * y_local[i];
        }

        alpha = -(r_local[k] + sum) / beta;

        for (i = 0; i < k; i++) {
#pragma HLS PIPELINE II=1
            z[i] = y_local[i] + alpha * y_local[k - i - 1];
        }

        for (i = 0; i < k; i++) {
#pragma HLS PIPELINE II=1
            y_local[i] = z[i];
        }

        y_local[k] = alpha;
    }

    // Write y_local back to output
    for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        y[i] = y_local[i];
    }
}
}