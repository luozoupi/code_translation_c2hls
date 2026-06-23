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
#pragma HLS INTERFACE s_axilite port=beta  bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int ni = NI;
    const int nj = NJ;
    const int nk = NK;

    int i, j, k;

    // Local buffers to stage the working set for reuse and parallel access.
    double a_row[NK];
    double b_buf[NK][NJ];
    double c_row[NJ];
#pragma HLS ARRAY_PARTITION variable=a_row cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=b_buf cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=c_row cyclic factor=8 dim=1

    // Stage B once (reused across all rows of C).
    for (k = 0; k < nk; k++) {
        for (j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
            b_buf[k][j] = B[k][j];
        }
    }

    for (i = 0; i < ni; i++) {
        // Load and scale current C row by beta.
        for (j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
            c_row[j] = C[i][j] * beta;
        }

        // Load current A row.
        for (k = 0; k < nk; k++) {
#pragma HLS PIPELINE II=1
            a_row[k] = A[i][k];
        }

        // Accumulate the matrix product into the local C row.
        for (k = 0; k < nk; k++) {
            double a_val = alpha * a_row[k];
            for (j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
                c_row[j] += a_val * b_buf[k][j];
            }
        }

        // Write the C row back.
        for (j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
            C[i][j] = c_row[j];
        }
    }
}