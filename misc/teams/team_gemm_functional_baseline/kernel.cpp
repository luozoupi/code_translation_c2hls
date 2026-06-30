// Team functional Phase B baseline for hlsfactory_gemm (from provided messages JSON).
// Claude Sonnet 4-6: m_axi interfaces only, no PIPELINE/UNROLL/PARTITION.
// Source: team handoff conversation (functional translate prompt + assistant reply).

#include "gemm.h"

extern "C" {

void kernel_gemm(
    double alpha,
    double beta,
    double C[60 + 0][70 + 0],
    double A[60 + 0][80 + 0],
    double B[80 + 0][70 + 0])
{
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=alpha bundle=control
#pragma HLS INTERFACE s_axilite port=beta bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int ni = 60;
    const int nj = 70;
    const int nk = 80;

    int i, j, k;
    for (i = 0; i < ni; i++) {
        for (j = 0; j < nj; j++)
            C[i][j] *= beta;
        for (k = 0; k < nk; k++) {
            for (j = 0; j < nj; j++)
                C[i][j] += alpha * A[i][k] * B[k][j];
        }
    }
}

} // extern "C"
