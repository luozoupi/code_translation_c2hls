#include "3mm.h"

extern "C" {

void kernel_3mm(    
		double E[ NI + 0][NJ + 0],
		double A[ NI + 0][NK + 0],
		double B[ NK + 0][NJ + 0],
		double F[ NJ + 0][NL + 0],
		double C[ NJ + 0][NM + 0],
		double D[ NM + 0][NL + 0],
		double G[ NI + 0][NL + 0])
{
#pragma HLS INTERFACE m_axi port=E offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=F offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=D offset=slave bundle=gmem5
#pragma HLS INTERFACE m_axi port=G offset=slave bundle=gmem6
#pragma HLS INTERFACE s_axilite port=E bundle=control
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=F bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=D bundle=control
#pragma HLS INTERFACE s_axilite port=G bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local buffers for tiling/reuse
    double localA[NI][NK];
    double localB[NK][NJ];
    double localC[NJ][NM];
    double localD[NM][NL];
    double localE[NI][NJ];
    double localF[NJ][NL];
    double localG[NI][NL];

#pragma HLS ARRAY_PARTITION variable=localA cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=localB cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=localC cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=localD cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=localE cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=localF cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=localG cyclic factor=8 dim=2

    // Load A
    load_A: for (int i = 0; i < NI; i++)
        for (int k = 0; k < NK; k++) {
#pragma HLS PIPELINE II=1
            localA[i][k] = A[i][k];
        }

    // Load B
    load_B: for (int k = 0; k < NK; k++)
        for (int j = 0; j < NJ; j++) {
#pragma HLS PIPELINE II=1
            localB[k][j] = B[k][j];
        }

    // Compute E = A * B
    comp_E_i: for (int i = 0; i < NI; i++) {
        comp_E_j: for (int j = 0; j < NJ; j++) {
#pragma HLS PIPELINE II=1
            double sum = 0.0;
            comp_E_k: for (int k = 0; k < NK; ++k) {
#pragma HLS UNROLL factor=8
                sum += localA[i][k] * localB[k][j];
            }
            localE[i][j] = sum;
        }
    }

    // Load C
    load_C: for (int i = 0; i < NJ; i++)
        for (int k = 0; k < NM; k++) {
#pragma HLS PIPELINE II=1
            localC[i][k] = C[i][k];
        }

    // Load D
    load_D: for (int k = 0; k < NM; k++)
        for (int j = 0; j < NL; j++) {
#pragma HLS PIPELINE II=1
            localD[k][j] = D[k][j];
        }

    // Compute F = C * D
    comp_F_i: for (int i = 0; i < NJ; i++) {
        comp_F_j: for (int j = 0; j < NL; j++) {
#pragma HLS PIPELINE II=1
            double sum = 0.0;
            comp_F_k: for (int k = 0; k < NM; ++k) {
#pragma HLS UNROLL factor=8
                sum += localC[i][k] * localD[k][j];
            }
            localF[i][j] = sum;
        }
    }

    // Compute G = E * F
    comp_G_i: for (int i = 0; i < NI; i++) {
        comp_G_j: for (int j = 0; j < NL; j++) {
#pragma HLS PIPELINE II=1
            double sum = 0.0;
            comp_G_k: for (int k = 0; k < NJ; ++k) {
#pragma HLS UNROLL factor=8
                sum += localE[i][k] * localF[k][j];
            }
            localG[i][j] = sum;
        }
    }

    // Write back E
    store_E: for (int i = 0; i < NI; i++)
        for (int j = 0; j < NJ; j++) {
#pragma HLS PIPELINE II=1
            E[i][j] = localE[i][j];
        }

    // Write back F
    store_F: for (int i = 0; i < NJ; i++)
        for (int j = 0; j < NL; j++) {
#pragma HLS PIPELINE II=1
            F[i][j] = localF[i][j];
        }

    // Write back G
    store_G: for (int i = 0; i < NI; i++)
        for (int j = 0; j < NL; j++) {
#pragma HLS PIPELINE II=1
            G[i][j] = localG[i][j];
        }
}

} // extern "C"