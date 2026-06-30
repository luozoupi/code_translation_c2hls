// Our in-repo gold / Phase-B functional reference (benchmarks/hlsfactory_gemm/hls_baseline.cpp).
// Naive loops + #pragma HLS top only (no explicit m_axi).

#include "gemm.h"

void kernel_gemm(
    double alpha,
    double beta,
    double C[NI + 0][NJ + 0],
    double A[NI + 0][NK + 0],
    double B[NK + 0][NJ + 0])
{
#pragma HLS top name=kernel_gemm

    const int ni = NI;
    const int nj = NJ;
    const int nk = NK;

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
