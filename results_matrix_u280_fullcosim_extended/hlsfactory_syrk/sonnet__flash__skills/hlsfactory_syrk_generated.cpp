#include "syrk.h"

extern "C" {

void kernel_syrk(
        double alpha,
        double beta,
        double C[N + 0][N + 0],
        double A[N + 0][M + 0])
{
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=alpha bundle=control
#pragma HLS INTERFACE s_axilite port=beta  bundle=control
#pragma HLS INTERFACE s_axilite port=C     bundle=control
#pragma HLS INTERFACE s_axilite port=A     bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;
    const int m = M;

    int i, j, k;

    for (i = 0; i < n; i++) {
        // Scale lower-triangular row i by beta
        for (j = 0; j <= i; j++) {
#pragma HLS PIPELINE II=1
            C[i][j] *= beta;
        }
        // Accumulate outer-product contributions — keep FP reduction serial
        for (k = 0; k < m; k++) {
            for (j = 0; j <= i; j++) {
#pragma HLS PIPELINE II=1
                C[i][j] += alpha * A[i][k] * A[j][k];
            }
        }
    }
}

} // extern "C"