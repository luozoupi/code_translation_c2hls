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

    // Local buffers for all matrices to enable partitioning and fast access
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
    load_A_i: for (int i = 0; i < NI; i++) {
        load_A_k: for (int k = 0; k < NK; k++) {
#pragma HLS PIPELINE II=1
            l_A[i][k] = A[i][k];
        }
    }

    // Load B
    load_B_k: for (int k = 0; k < NK; k++) {
        load_B_j: for (int j = 0; j < NJ; j++) {
#pragma HLS PIPELINE II=1
            l_B[k][j] = B[k][j];
        }
    }

    // Load C
    load_C_j: for (int j = 0; j < NJ; j++) {
        load_C_l: for (int l = 0; l < NL; l++) {
#pragma HLS PIPELINE II=1
            l_C[j][l] = C[j][l];
        }
    }

    // Load D
    load_D_i: for (int i = 0; i < NI; i++) {
        load_D_l: for (int l = 0; l < NL; l++) {
#pragma HLS PIPELINE II=1
            l_D[i][l] = D[i][l];
        }
    }

    // First matrix multiply: tmp = alpha * A * B
    mm1_i: for (int i = 0; i < NI; i++) {
        mm1_j: for (int j = 0; j < NJ; j++) {
#pragma HLS PIPELINE II=1
            double acc = 0.0;
            mm1_k: for (int k = 0; k < NK; k++) {
#pragma HLS UNROLL factor=8
                acc += alpha * l_A[i][k] * l_B[k][j];
            }
            l_tmp[i][j] = acc;
        }
    }

    // Second matrix multiply: D = beta * D + tmp * C
    mm2_i: for (int i = 0; i < NI; i++) {
        mm2_j: for (int j = 0; j < NL; j++) {
#pragma HLS PIPELINE II=1
            double acc = beta * l_D[i][j];
            mm2_k: for (int k = 0; k < NJ; k++) {
#pragma HLS UNROLL factor=8
                acc += l_tmp[i][k] * l_C[k][j];
            }
            l_D[i][j] = acc;
        }
    }

    // Write back tmp
    store_tmp_i: for (int i = 0; i < NI; i++) {
        store_tmp_j: for (int j = 0; j < NJ; j++) {
#pragma HLS PIPELINE II=1
            tmp[i][j] = l_tmp[i][j];
        }
    }

    // Write back D
    store_D_i: for (int i = 0; i < NI; i++) {
        store_D_l: for (int l = 0; l < NL; l++) {
#pragma HLS PIPELINE II=1
            D[i][l] = l_D[i][l];
        }
    }
}

} // extern "C"