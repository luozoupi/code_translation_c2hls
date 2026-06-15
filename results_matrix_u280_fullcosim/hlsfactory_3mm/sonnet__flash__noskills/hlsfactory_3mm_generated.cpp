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

    const int ni = NI;
    const int nj = NJ;
    const int nk = NK;
    const int nl = NL;
    const int nm = NM;

    // Local buffers for better memory access patterns
    double E_local[NI][NJ];
    double A_local[NI][NK];
    double B_local[NK][NJ];
    double F_local[NJ][NL];
    double C_local[NJ][NM];
    double D_local[NM][NL];
    double G_local[NI][NL];

#pragma HLS ARRAY_PARTITION variable=E_local cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=A_local cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=B_local cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=F_local cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=C_local cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=D_local cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=G_local cyclic factor=8 dim=2

    // Load A
    load_A: for (int i = 0; i < ni; i++)
        for (int j = 0; j < nk; j++) {
#pragma HLS PIPELINE II=1
            A_local[i][j] = A[i][j];
        }

    // Load B
    load_B: for (int i = 0; i < nk; i++)
        for (int j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
            B_local[i][j] = B[i][j];
        }

    // Load C
    load_C: for (int i = 0; i < nj; i++)
        for (int j = 0; j < nm; j++) {
#pragma HLS PIPELINE II=1
            C_local[i][j] = C[i][j];
        }

    // Load D
    load_D: for (int i = 0; i < nm; i++)
        for (int j = 0; j < nl; j++) {
#pragma HLS PIPELINE II=1
            D_local[i][j] = D[i][j];
        }

    int i, j, k;

    // E = A * B
    loop_E_i: for (i = 0; i < ni; i++) {
        loop_E_j: for (j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
            double sum = 0.0;
            loop_E_k: for (k = 0; k < nk; ++k) {
#pragma HLS UNROLL factor=4
                sum += A_local[i][k] * B_local[k][j];
            }
            E_local[i][j] = sum;
        }
    }

    // F = C * D
    loop_F_i: for (i = 0; i < nj; i++) {
        loop_F_j: for (j = 0; j < nl; j++) {
#pragma HLS PIPELINE II=1
            double sum = 0.0;
            loop_F_k: for (k = 0; k < nm; ++k) {
#pragma HLS UNROLL factor=4
                sum += C_local[i][k] * D_local[k][j];
            }
            F_local[i][j] = sum;
        }
    }

    // G = E * F
    loop_G_i: for (i = 0; i < ni; i++) {
        loop_G_j: for (j = 0; j < nl; j++) {
#pragma HLS PIPELINE II=1
            double sum = 0.0;
            loop_G_k: for (k = 0; k < nj; ++k) {
#pragma HLS UNROLL factor=4
                sum += E_local[i][k] * F_local[k][j];
            }
            G_local[i][j] = sum;
        }
    }

    // Write back E
    store_E: for (int i = 0; i < ni; i++)
        for (int j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
            E[i][j] = E_local[i][j];
        }

    // Write back F
    store_F: for (int i = 0; i < nj; i++)
        for (int j = 0; j < nl; j++) {
#pragma HLS PIPELINE II=1
            F[i][j] = F_local[i][j];
        }

    // Write back G
    store_G: for (int i = 0; i < ni; i++)
        for (int j = 0; j < nl; j++) {
#pragma HLS PIPELINE II=1
            G[i][j] = G_local[i][j];
        }
}

} // extern "C"