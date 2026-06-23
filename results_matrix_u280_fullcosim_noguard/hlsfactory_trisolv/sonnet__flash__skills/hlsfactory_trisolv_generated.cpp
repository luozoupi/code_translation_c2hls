#include "trisolv.h"

extern "C" {

void kernel_trisolv(
        double L[N + 0][N + 0],
        double x[N + 0],
        double b[N + 0])
{
#pragma HLS INTERFACE m_axi port=L offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=L bundle=control
#pragma HLS INTERFACE s_axilite port=x bundle=control
#pragma HLS INTERFACE s_axilite port=b bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local buffers for x and b to allow parallel access
    double x_local[N];
    double b_local[N];
    double L_row[N];

#pragma HLS ARRAY_PARTITION variable=x_local cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=L_row cyclic factor=8 dim=1

    const int n = N;

    // Load b into local buffer
    load_b: for (int k = 0; k < n; k++) {
#pragma HLS PIPELINE II=1
        b_local[k] = b[k];
    }

    int i, j;

    for (i = 0; i < n; i++) {
        // Load L row i into local buffer
        load_L_row: for (int k = 0; k < n; k++) {
#pragma HLS PIPELINE II=1
            L_row[k] = L[i][k];
        }

        double xi = b_local[i];

        // Inner loop: accumulate sum L[i][j] * x[j] for j < i
        inner_loop: for (j = 0; j < i; j++) {
#pragma HLS PIPELINE II=1
            xi -= L_row[j] * x_local[j];
        }

        xi = xi / L_row[i];
        x_local[i] = xi;
    }

    // Write x back
    store_x: for (int k = 0; k < n; k++) {
#pragma HLS PIPELINE II=1
        x[k] = x_local[k];
    }
}

} // extern "C"