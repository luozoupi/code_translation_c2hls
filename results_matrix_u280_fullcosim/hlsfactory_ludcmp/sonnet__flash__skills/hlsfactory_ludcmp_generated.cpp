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

    // Local copies for efficient on-chip access
    double lA[N][N];
    double lb[N];
    double lx[N];
    double ly[N];

#pragma HLS ARRAY_PARTITION variable=lA cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=lb complete dim=1
#pragma HLS ARRAY_PARTITION variable=lx complete dim=1
#pragma HLS ARRAY_PARTITION variable=ly complete dim=1

    const int n = N;
    int i, j, k;
    double w;

    // Load A from global memory
    load_A_outer: for (i = 0; i < n; i++) {
        load_A_inner: for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            lA[i][j] = A[i][j];
        }
    }

    // Load b
    load_b: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        lb[i] = b[i];
    }

    // Load y (used as accumulator in forward substitution)
    load_y: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        ly[i] = y[i];
    }

    // Load x (used as accumulator in backward substitution)
    load_x: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        lx[i] = x[i];
    }

    // LU Factorization
    lu_outer: for (i = 0; i < n; i++) {
        // Lower triangular part
        lu_lower_j: for (j = 0; j < i; j++) {
            w = lA[i][j];
            lu_lower_k: for (k = 0; k < j; k++) {
#pragma HLS PIPELINE II=1
                w -= lA[i][k] * lA[k][j];
            }
            lA[i][j] = w / lA[j][j];
        }
        // Upper triangular part
        lu_upper_j: for (j = i; j < n; j++) {
            w = lA[i][j];
            lu_upper_k: for (k = 0; k < i; k++) {
#pragma HLS PIPELINE II=1
                w -= lA[i][k] * lA[k][j];
            }
            lA[i][j] = w;
        }
    }

    // Forward substitution: solve L*y = b
    fwd_outer: for (i = 0; i < n; i++) {
        w = lb[i];
        fwd_inner: for (j = 0; j < i; j++) {
#pragma HLS PIPELINE II=1
            w -= lA[i][j] * ly[j];
        }
        ly[i] = w;
    }

    // Backward substitution: solve U*x = y
    bwd_outer: for (i = n-1; i >= 0; i--) {
        w = ly[i];
        bwd_inner: for (j = i+1; j < n; j++) {
#pragma HLS PIPELINE II=1
            w -= lA[i][j] * lx[j];
        }
        lx[i] = w / lA[i][i];
    }

    // Store A back to global memory
    store_A_outer: for (i = 0; i < n; i++) {
        store_A_inner: for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            A[i][j] = lA[i][j];
        }
    }

    // Store y back
    store_y: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        y[i] = ly[i];
    }

    // Store x back
    store_x: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        x[i] = lx[i];
    }
}

} // extern "C"