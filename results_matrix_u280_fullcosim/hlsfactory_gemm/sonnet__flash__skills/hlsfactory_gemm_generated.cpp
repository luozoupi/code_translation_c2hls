#include "gemm.h"

extern "C" {

void kernel_gemm(
     double alpha,
     double beta,
     double C[ NI + 0][NJ + 0],
     double A[ NI + 0][NK + 0],
     double B[ NK + 0][NJ + 0])
{
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=alpha bundle=control
#pragma HLS INTERFACE s_axilite port=beta bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local tile buffers for staging
    double local_C[NI][NJ];
    double local_A[NI][NK];
    double local_B[NK][NJ];

#pragma HLS ARRAY_PARTITION variable=local_C cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=local_A cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=local_B cyclic factor=8 dim=2

    // Load C into local buffer
    load_C_i: for (int i = 0; i < NI; i++) {
        load_C_j: for (int j = 0; j < NJ; j++) {
#pragma HLS PIPELINE II=1
            local_C[i][j] = C[i][j];
        }
    }

    // Load A into local buffer
    load_A_i: for (int i = 0; i < NI; i++) {
        load_A_k: for (int k = 0; k < NK; k++) {
#pragma HLS PIPELINE II=1
            local_A[i][k] = A[i][k];
        }
    }

    // Load B into local buffer
    load_B_k: for (int k = 0; k < NK; k++) {
        load_B_j: for (int j = 0; j < NJ; j++) {
#pragma HLS PIPELINE II=1
            local_B[k][j] = B[k][j];
        }
    }

    // Compute: scale C by beta, then accumulate alpha * A * B
    const int ni = NI;
    const int nj = NJ;
    const int nk = NK;

    int i, j, k;
    for (i = 0; i < ni; i++) {
        // Scale row i of C by beta
        for (j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
            local_C[i][j] *= beta;
        }
        // Accumulate A[i][k] * B[k][j] scaled by alpha
        for (k = 0; k < nk; k++) {
            double a_val = local_A[i][k];
            for (j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
                local_C[i][j] += alpha * a_val * local_B[k][j];
            }
        }
    }

    // Store result back to global memory
    store_C_i: for (int i = 0; i < NI; i++) {
        store_C_j: for (int j = 0; j < NJ; j++) {
#pragma HLS PIPELINE II=1
            C[i][j] = local_C[i][j];
        }
    }
}

} // extern "C"