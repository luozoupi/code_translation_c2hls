#include "gemm.h"

extern "C" {

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
#pragma HLS INTERFACE s_axilite port=beta  bundle=control
#pragma HLS INTERFACE s_axilite port=C     bundle=control
#pragma HLS INTERFACE s_axilite port=A     bundle=control
#pragma HLS INTERFACE s_axilite port=B     bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local copies to avoid repeated global-memory traffic and enable
    // partitioning for pipelined access
    double l_C[NI][NJ];
    double l_A[NI][NK];
    double l_B[NK][NJ];

#pragma HLS ARRAY_PARTITION variable=l_C cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_A cyclic factor=1 dim=2
#pragma HLS ARRAY_PARTITION variable=l_B cyclic factor=8 dim=2

    // Load C
    for (int i = 0; i < NI; i++) {
        for (int j = 0; j < NJ; j++) {
#pragma HLS PIPELINE II=1
            l_C[i][j] = C[i][j];
        }
    }

    // Load A
    for (int i = 0; i < NI; i++) {
        for (int k = 0; k < NK; k++) {
#pragma HLS PIPELINE II=1
            l_A[i][k] = A[i][k];
        }
    }

    // Load B
    for (int k = 0; k < NK; k++) {
        for (int j = 0; j < NJ; j++) {
#pragma HLS PIPELINE II=1
            l_B[k][j] = B[k][j];
        }
    }

    const int ni = NI;
    const int nj = NJ;
    const int nk = NK;

    int i, j, k;

    // Compute: scale C by beta, then accumulate alpha*A*B
    for (i = 0; i < ni; i++) {
        // Scale C[i][j] *= beta
        for (j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
            l_C[i][j] *= beta;
        }
        // Accumulate: C[i][j] += alpha * A[i][k] * B[k][j]
        // k is the reduction dimension — do NOT unroll k (FP reduction order must be preserved)
        for (k = 0; k < nk; k++) {
            double a_ik = l_A[i][k];
            for (j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
                l_C[i][j] += alpha * a_ik * l_B[k][j];
            }
        }
    }

    // Store C
    for (int i = 0; i < NI; i++) {
        for (int j = 0; j < NJ; j++) {
#pragma HLS PIPELINE II=1
            C[i][j] = l_C[i][j];
        }
    }
}

} // extern "C"