#include "2mm.h"

extern "C" {

void kernel_2mm(
        double alpha,
        double beta,
        double tmp[ NI + 0][NJ + 0],
        double A[ NI + 0][NK + 0],
        double B[ NK + 0][NJ + 0],
        double C[ NJ + 0][NL + 0],
        double D[ NI + 0][NL + 0])
{
#pragma HLS INTERFACE m_axi port=tmp offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=A   offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=B   offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=C   offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=D   offset=slave bundle=gmem4
#pragma HLS INTERFACE s_axilite port=alpha  bundle=control
#pragma HLS INTERFACE s_axilite port=beta   bundle=control
#pragma HLS INTERFACE s_axilite port=tmp    bundle=control
#pragma HLS INTERFACE s_axilite port=A      bundle=control
#pragma HLS INTERFACE s_axilite port=B      bundle=control
#pragma HLS INTERFACE s_axilite port=C      bundle=control
#pragma HLS INTERFACE s_axilite port=D      bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local buffers for better memory access patterns
    double l_tmp[NI][NJ];
    double l_A[NI][NK];
    double l_B[NK][NJ];
    double l_C[NJ][NL];
    double l_D[NI][NL];

#pragma HLS ARRAY_PARTITION variable=l_tmp cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_A   cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_B   cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_C   cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_D   cyclic factor=8 dim=2

    // Load A
    for (int i = 0; i < NI; i++)
        for (int k = 0; k < NK; k++) {
#pragma HLS PIPELINE II=1
            l_A[i][k] = A[i][k];
        }

    // Load B
    for (int k = 0; k < NK; k++)
        for (int j = 0; j < NJ; j++) {
#pragma HLS PIPELINE II=1
            l_B[k][j] = B[k][j];
        }

    // Load C
    for (int j = 0; j < NJ; j++)
        for (int l = 0; l < NL; l++) {
#pragma HLS PIPELINE II=1
            l_C[j][l] = C[j][l];
        }

    // Load D
    for (int i = 0; i < NI; i++)
        for (int l = 0; l < NL; l++) {
#pragma HLS PIPELINE II=1
            l_D[i][l] = D[i][l];
        }

    // Compute tmp = alpha * A * B
    for (int i = 0; i < NI; i++) {
        for (int j = 0; j < NJ; j++) {
#pragma HLS PIPELINE II=1
            double sum = 0.0;
            for (int k = 0; k < NK; k++) {
#pragma HLS UNROLL factor=8
                sum += alpha * l_A[i][k] * l_B[k][j];
            }
            l_tmp[i][j] = sum;
        }
    }

    // Compute D = beta * D + tmp * C
    for (int i = 0; i < NI; i++) {
        for (int j = 0; j < NL; j++) {
#pragma HLS PIPELINE II=1
            double sum = l_D[i][j] * beta;
            for (int k = 0; k < NJ; k++) {
#pragma HLS UNROLL factor=8
                sum += l_tmp[i][k] * l_C[k][j];
            }
            l_D[i][j] = sum;
        }
    }

    // Write back tmp
    for (int i = 0; i < NI; i++)
        for (int j = 0; j < NJ; j++) {
#pragma HLS PIPELINE II=1
            tmp[i][j] = l_tmp[i][j];
        }

    // Write back D
    for (int i = 0; i < NI; i++)
        for (int l = 0; l < NL; l++) {
#pragma HLS PIPELINE II=1
            D[i][l] = l_D[i][l];
        }
}

} // extern "C"