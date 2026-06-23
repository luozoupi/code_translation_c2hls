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
#pragma HLS INTERFACE s_axilite port=beta bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int ni = NI;
    const int nj = NJ;
    const int nk = NK;

    int i, j, k;

    // Local buffer for B (reused across all rows of C)
    static double Bbuf[NK][NJ];
#pragma HLS ARRAY_PARTITION variable=Bbuf cyclic factor=8 dim=2

    // Load B once into local memory
    for (k = 0; k < nk; k++) {
        for (j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
            Bbuf[k][j] = B[k][j];
        }
    }

    for (i = 0; i < ni; i++) {
        // Local row buffers
        double Crow[NJ];
        double Arow[NK];
#pragma HLS ARRAY_PARTITION variable=Crow cyclic factor=8 dim=1

        // Load A row
        for (k = 0; k < nk; k++) {
#pragma HLS PIPELINE II=1
            Arow[k] = A[i][k];
        }

        // Scale C row by beta
        for (j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
            Crow[j] = C[i][j] * beta;
        }

        // Accumulate: iterate over k outer, j inner.
        // Inner loop over j has no inter-iteration dependence -> II=1
        for (k = 0; k < nk; k++) {
            double a_alpha = alpha * Arow[k];
            for (j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
                Crow[j] += a_alpha * Bbuf[k][j];
            }
        }

        // Write C row back
        for (j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
            C[i][j] = Crow[j];
        }
    }
}
}