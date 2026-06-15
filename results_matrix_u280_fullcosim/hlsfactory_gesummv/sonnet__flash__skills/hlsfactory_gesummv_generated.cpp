#include "gesummv.h"

extern "C" {

void kernel_gesummv(
        double alpha,
        double beta,
        double A[ N + 0][N + 0],
        double B[ N + 0][N + 0],
        double tmp[ N + 0],
        double x[ N + 0],
        double y[ N + 0])
{
#pragma HLS INTERFACE m_axi port=A   offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=B   offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=tmp offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=x   offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=y   offset=slave bundle=gmem4
#pragma HLS INTERFACE s_axilite port=alpha  bundle=control
#pragma HLS INTERFACE s_axilite port=beta   bundle=control
#pragma HLS INTERFACE s_axilite port=A      bundle=control
#pragma HLS INTERFACE s_axilite port=B      bundle=control
#pragma HLS INTERFACE s_axilite port=tmp    bundle=control
#pragma HLS INTERFACE s_axilite port=x      bundle=control
#pragma HLS INTERFACE s_axilite port=y      bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local buffers to allow partitioning and avoid repeated global memory reads
    double lA[N][N];
    double lB[N][N];
    double lx[N];
    double ltmp[N];
    double ly[N];

#pragma HLS ARRAY_PARTITION variable=lA   cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=lB   cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=lx   cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=ltmp complete dim=1
#pragma HLS ARRAY_PARTITION variable=ly   complete dim=1

    // Load x from global memory
    load_x: for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
        lx[j] = x[j];
    }

    // Load A from global memory
    load_A: for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            lA[i][j] = A[i][j];
        }
    }

    // Load B from global memory
    load_B: for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            lB[i][j] = B[i][j];
        }
    }

    const int n = N;
    int i, j;

    // Main computation
    for (i = 0; i < n; i++) {
        ltmp[i] = 0.0;
        ly[i]   = 0.0;
        for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            ltmp[i] = lA[i][j] * lx[j] + ltmp[i];
            ly[i]   = lB[i][j] * lx[j] + ly[i];
        }
        ly[i] = alpha * ltmp[i] + beta * ly[i];
    }

    // Write tmp and y back to global memory
    store_tmp: for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        tmp[i] = ltmp[i];
    }

    store_y: for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        y[i] = ly[i];
    }
}

} // extern "C"