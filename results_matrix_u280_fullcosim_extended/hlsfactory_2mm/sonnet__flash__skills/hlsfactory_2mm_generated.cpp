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
#pragma HLS INTERFACE m_axi port=tmp    offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=A      offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=B      offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=C      offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=D      offset=slave bundle=gmem4
#pragma HLS INTERFACE s_axilite port=alpha  bundle=control
#pragma HLS INTERFACE s_axilite port=beta   bundle=control
#pragma HLS INTERFACE s_axilite port=tmp    bundle=control
#pragma HLS INTERFACE s_axilite port=A      bundle=control
#pragma HLS INTERFACE s_axilite port=B      bundle=control
#pragma HLS INTERFACE s_axilite port=C      bundle=control
#pragma HLS INTERFACE s_axilite port=D      bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int ni = NI;
    const int nj = NJ;
    const int nk = NK;
    const int nl = NL;

    // Local buffers to avoid repeated global memory accesses
    double local_tmp[NI][NJ];
    double local_A[NI][NK];
    double local_B[NK][NJ];
    double local_C[NJ][NL];
    double local_D[NI][NL];

#pragma HLS ARRAY_PARTITION variable=local_tmp cyclic factor=2 dim=2
#pragma HLS ARRAY_PARTITION variable=local_A   cyclic factor=2 dim=2
#pragma HLS ARRAY_PARTITION variable=local_B   cyclic factor=2 dim=2
#pragma HLS ARRAY_PARTITION variable=local_C   cyclic factor=2 dim=2
#pragma HLS ARRAY_PARTITION variable=local_D   cyclic factor=2 dim=2

    // Load A
    load_A: for (int i = 0; i < NI; i++)
        for (int k = 0; k < NK; k++) {
#pragma HLS PIPELINE II=1
            local_A[i][k] = A[i][k];
        }

    // Load B
    load_B: for (int k = 0; k < NK; k++)
        for (int j = 0; j < NJ; j++) {
#pragma HLS PIPELINE II=1
            local_B[k][j] = B[k][j];
        }

    // Compute tmp = alpha * A * B  (preserve serial FP reduction order)
    loop1_i: for (int i = 0; i < ni; i++) {
        loop1_j: for (int j = 0; j < nj; j++) {
            double acc = 0.0;
            loop1_k: for (int k = 0; k < nk; ++k) {
#pragma HLS PIPELINE II=1
                acc += alpha * local_A[i][k] * local_B[k][j];
            }
            local_tmp[i][j] = acc;
        }
    }

    // Store tmp back
    store_tmp: for (int i = 0; i < NI; i++)
        for (int j = 0; j < NJ; j++) {
#pragma HLS PIPELINE II=1
            tmp[i][j] = local_tmp[i][j];
        }

    // Load C
    load_C: for (int j = 0; j < NJ; j++)
        for (int l = 0; l < NL; l++) {
#pragma HLS PIPELINE II=1
            local_C[j][l] = C[j][l];
        }

    // Load D
    load_D: for (int i = 0; i < NI; i++)
        for (int l = 0; l < NL; l++) {
#pragma HLS PIPELINE II=1
            local_D[i][l] = D[i][l];
        }

    // Compute D = beta*D + tmp*C  (preserve serial FP reduction order)
    loop2_i: for (int i = 0; i < ni; i++) {
        loop2_j: for (int j = 0; j < nl; j++) {
            double acc = local_D[i][j] * beta;
            loop2_k: for (int k = 0; k < nj; ++k) {
#pragma HLS PIPELINE II=1
                acc += local_tmp[i][k] * local_C[k][j];
            }
            local_D[i][j] = acc;
        }
    }

    // Store D back
    store_D: for (int i = 0; i < NI; i++)
        for (int l = 0; l < NL; l++) {
#pragma HLS PIPELINE II=1
            D[i][l] = local_D[i][l];
        }
}

} // extern "C"