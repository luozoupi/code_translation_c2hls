#include "gemm.h"


void kernel_gemm(  
		 double alpha,
		 double beta,
		 double C[ NI + 0][NJ + 0],
		 double A[ NI + 0][NK + 0],
		 double B[ NK + 0][NJ + 0])
{
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=alpha bundle=control
#pragma HLS INTERFACE s_axilite port=beta bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int ni = NI;
    const int nj = NJ;
    const int nk = NK;

    // Local staging buffers to enable reuse and parallel access.
    double A_buf[NI][NK];
    double B_buf[NK][NJ];
    double C_buf[NI][NJ];
#pragma HLS ARRAY_PARTITION variable=A_buf cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=B_buf cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=C_buf cyclic factor=8 dim=2

    int i, j, k;

    // Load A into local buffer.
load_A_i:
    for (i = 0; i < ni; i++) {
    load_A_k:
        for (k = 0; k < nk; k++) {
#pragma HLS PIPELINE II=1
            A_buf[i][k] = A[i][k];
        }
    }

    // Load B into local buffer.
load_B_k:
    for (k = 0; k < nk; k++) {
    load_B_j:
        for (j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
            B_buf[k][j] = B[k][j];
        }
    }

    // Load C into local buffer.
load_C_i:
    for (i = 0; i < ni; i++) {
    load_C_j:
        for (j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
            C_buf[i][j] = C[i][j];
        }
    }

    // Compute on local buffers.
compute_i:
    for (i = 0; i < ni; i++) {
    scale_j:
        for (j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
            C_buf[i][j] *= beta;
        }
    compute_k:
        for (k = 0; k < nk; k++) {
        compute_j:
            for (j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
                C_buf[i][j] += alpha * A_buf[i][k] * B_buf[k][j];
            }
        }
    }

    // Store C back to global memory.
store_C_i:
    for (i = 0; i < ni; i++) {
    store_C_j:
        for (j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
            C[i][j] = C_buf[i][j];
        }
    }
}