#include "trisolv.h"

extern "C" {

void kernel_trisolv(
		    double L[ N + 0][N + 0],
		    double x[ N + 0],
		    double b[ N + 0])
{
#pragma HLS INTERFACE m_axi port=L offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=L bundle=control
#pragma HLS INTERFACE s_axilite port=x bundle=control
#pragma HLS INTERFACE s_axilite port=b bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    double x_local[N];
    double b_local[N];
    double L_local[N][N];

#pragma HLS ARRAY_PARTITION variable=x_local cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=L_local cyclic factor=8 dim=2

    // Load b into local buffer
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        b_local[i] = b[i];
    }

    // Load L into local buffer
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            L_local[i][j] = L[i][j];
        }
    }

    const int n = N;
    int i, j;

    for (i = 0; i < n; i++) {
        x_local[i] = b_local[i];
        for (j = 0; j < i; j++) {
#pragma HLS PIPELINE II=1
            x_local[i] -= L_local[i][j] * x_local[j];
        }
        x_local[i] = x_local[i] / L_local[i][i];
    }

    // Write results back
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        x[i] = x_local[i];
    }
}

} // extern "C"