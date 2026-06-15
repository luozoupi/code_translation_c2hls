#include "ludcmp.h"

extern "C" {

void kernel_ludcmp(
		   double A[ N + 0][N + 0],
		   double b[ N + 0],
		   double x[ N + 0],
		   double y[ N + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=y offset=slave bundle=gmem3
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=b bundle=control
#pragma HLS INTERFACE s_axilite port=x bundle=control
#pragma HLS INTERFACE s_axilite port=y bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local copies to enable fast on-chip access and partitioning
    double lA[N][N];
    double lb[N];
    double lx[N];
    double ly[N];

#pragma HLS ARRAY_PARTITION variable=lA cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=lb cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=lx cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=ly cyclic factor=8 dim=1

    // Load A from global memory
    load_A_outer: for (int i = 0; i < N; i++) {
        load_A_inner: for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            lA[i][j] = A[i][j];
        }
    }

    // Load b
    load_b: for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        lb[i] = b[i];
    }

    // Load x
    load_x: for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        lx[i] = x[i];
    }

    // Load y
    load_y: for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        ly[i] = y[i];
    }

    const int n = N;
    int i, j, k;
    double w;

    // LU decomposition
    for (i = 0; i < n; i++) {
        for (j = 0; j < i; j++) {
            w = lA[i][j];
            for (k = 0; k < j; k++) {
#pragma HLS PIPELINE II=1
                w -= lA[i][k] * lA[k][j];
            }
            lA[i][j] = w / lA[j][j];
        }
        for (j = i; j < n; j++) {
            w = lA[i][j];
            for (k = 0; k < i; k++) {
#pragma HLS PIPELINE II=1
                w -= lA[i][k] * lA[k][j];
            }
            lA[i][j] = w;
        }
    }

    // Forward substitution: solve L*y = b
    for (i = 0; i < n; i++) {
        w = lb[i];
        for (j = 0; j < i; j++) {
#pragma HLS PIPELINE II=1
            w -= lA[i][j] * ly[j];
        }
        ly[i] = w;
    }

    // Backward substitution: solve U*x = y
    for (i = n-1; i >= 0; i--) {
        w = ly[i];
        for (j = i+1; j < n; j++) {
#pragma HLS PIPELINE II=1
            w -= lA[i][j] * lx[j];
        }
        lx[i] = w / lA[i][i];
    }

    // Write back A
    store_A_outer: for (int i = 0; i < N; i++) {
        store_A_inner: for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            A[i][j] = lA[i][j];
        }
    }

    // Write back b
    store_b: for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        b[i] = lb[i];
    }

    // Write back x
    store_x: for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        x[i] = lx[i];
    }

    // Write back y
    store_y: for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        y[i] = ly[i];
    }
}

} // extern "C"