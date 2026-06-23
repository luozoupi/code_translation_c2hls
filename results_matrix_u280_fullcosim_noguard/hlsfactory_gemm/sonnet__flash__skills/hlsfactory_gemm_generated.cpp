#include "gemm.h"

extern "C" {

void kernel_gemm(
         double alpha,
         double beta,
         double C[ NI + 0][NJ + 0],
         double A[ NI + 0][NK + 0],
         double B[ NK + 0][NJ + 0])
{
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=alpha bundle=control
#pragma HLS INTERFACE s_axilite port=beta  bundle=control
#pragma HLS INTERFACE s_axilite port=C     bundle=control
#pragma HLS INTERFACE s_axilite port=A     bundle=control
#pragma HLS INTERFACE s_axilite port=B     bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local tile buffers for data reuse
    double localA[NI][NK];
    double localB[NK][NJ];
    double localC[NI][NJ];

#pragma HLS ARRAY_PARTITION variable=localA cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=localB cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=localC cyclic factor=8 dim=2

    // Load A from global memory
    load_A_i: for (int i = 0; i < NI; i++) {
        load_A_k: for (int k = 0; k < NK; k++) {
#pragma HLS PIPELINE II=1
            localA[i][k] = A[i][k];
        }
    }

    // Load B from global memory
    load_B_k: for (int k = 0; k < NK; k++) {
        load_B_j: for (int j = 0; j < NJ; j++) {
#pragma HLS PIPELINE II=1
            localB[k][j] = B[k][j];
        }
    }

    // Load C from global memory
    load_C_i: for (int i = 0; i < NI; i++) {
        load_C_j: for (int j = 0; j < NJ; j++) {
#pragma HLS PIPELINE II=1
            localC[i][j] = C[i][j];
        }
    }

    // Compute: scale C by beta, then accumulate alpha*A*B
    int i, j, k;
    for (i = 0; i < NI; i++) {
        // Scale row of C by beta
        scale_j: for (j = 0; j < NJ; j++) {
#pragma HLS PIPELINE II=1
            localC[i][j] *= beta;
        }
        // Accumulate alpha * A[i][k] * B[k][j] into C[i][j]
        for (k = 0; k < NK; k++) {
            accum_j: for (j = 0; j < NJ; j++) {
#pragma HLS PIPELINE II=1
                localC[i][j] += alpha * localA[i][k] * localB[k][j];
            }
        }
    }

    // Store C back to global memory
    store_C_i: for (int ii = 0; ii < NI; ii++) {
        store_C_j: for (int jj = 0; jj < NJ; jj++) {
#pragma HLS PIPELINE II=1
            C[ii][jj] = localC[ii][jj];
        }
    }
}

} // extern "C"