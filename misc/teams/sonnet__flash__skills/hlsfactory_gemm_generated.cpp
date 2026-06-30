#include "gemm.h"

extern "C" {

void kernel_gemm(
     double alpha,
     double beta,
     double C[60][70],
     double A[60][80],
     double B[80][70])
{
#pragma HLS INTERFACE s_axilite port=alpha    bundle=control
#pragma HLS INTERFACE s_axilite port=beta     bundle=control
#pragma HLS INTERFACE s_axilite port=C        bundle=control
#pragma HLS INTERFACE s_axilite port=A        bundle=control
#pragma HLS INTERFACE s_axilite port=B        bundle=control
#pragma HLS INTERFACE s_axilite port=return   bundle=control

    // Partition arrays along the column dimension to enable
    // parallel access during the pipelined j-loop
#pragma HLS ARRAY_PARTITION variable=C cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=A cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=B cyclic factor=4 dim=2

    const int ni = 60;
    const int nj = 70;
    const int nk = 80;

    int i, j, k;

    for (i = 0; i < ni; i++) {
        // Scale C[i][j] by beta
        for (j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
            C[i][j] *= beta;
        }
        // Accumulate alpha * A[i][k] * B[k][j] into C[i][j]
        for (k = 0; k < nk; k++) {
            for (j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
                C[i][j] += alpha * A[i][k] * B[k][j];
            }
        }
    }
}

} // extern "C"