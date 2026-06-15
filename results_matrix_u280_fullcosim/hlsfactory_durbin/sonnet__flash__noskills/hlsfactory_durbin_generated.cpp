#include "durbin.h"

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
    #pragma HLS ARRAY_PARTITION variable=z cyclic factor=4 dim=1

    double alpha;
    double beta;
    double sum;

    int i, k;

    y[0] = -r[0];
    beta = 1.0;
    alpha = -r[0];

    for (k = 1; k < n; k++) {
        beta = (1 - alpha * alpha) * beta;
        sum = 0.0;
        for (i = 0; i < k; i++) {
            #pragma HLS PIPELINE II=1
            sum += r[k - i - 1] * y[i];
        }
        alpha = -(r[k] + sum) / beta;

        for (i = 0; i < k; i++) {
            #pragma HLS PIPELINE II=1
            z[i] = y[i] + alpha * y[k - i - 1];
        }
        for (i = 0; i < k; i++) {
            #pragma HLS PIPELINE II=1
            y[i] = z[i];
        }
        y[k] = alpha;
    }
}