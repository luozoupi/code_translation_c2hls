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

    // Local buffers for intermediate results to enable partitioning
    static double E_local[NI][NJ];
    static double F_local[NJ][NL];

#pragma HLS ARRAY_PARTITION variable=E_local cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=F_local cyclic factor=8 dim=2

    const int ni = NI;
    const int nj = NJ;
    const int nk = NK;
    const int nl = NL;
    const int nm = NM;

    int i, j, k;

    // Compute E = A * B
    for (i = 0; i < ni; i++) {
        for (j = 0; j < nj; j++) {
            E_local[i][j] = 0.0;
        }
    }

    for (i = 0; i < ni; i++) {
        for (j = 0; j < nj; j++) {
            double sum = 0.0;
            for (k = 0; k < nk; ++k) {
#pragma HLS PIPELINE II=1
                sum += A[i][k] * B[k][j];
            }
            E_local[i][j] = sum;
        }
    }

    // Write E back to global memory
    for (i = 0; i < ni; i++) {
        for (j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
            E[i][j] = E_local[i][j];
        }
    }

    // Compute F = C * D
    for (i = 0; i < nj; i++) {
        for (j = 0; j < nl; j++) {
            F_local[i][j] = 0.0;
        }
    }

    for (i = 0; i < nj; i++) {
        for (j = 0; j < nl; j++) {
            double sum = 0.0;
            for (k = 0; k < nm; ++k) {
#pragma HLS PIPELINE II=1
                sum += C[i][k] * D[k][j];
            }
            F_local[i][j] = sum;
        }
    }

    // Write F back to global memory
    for (i = 0; i < nj; i++) {
        for (j = 0; j < nl; j++) {
#pragma HLS PIPELINE II=1
            F[i][j] = F_local[i][j];
        }
    }

    // Compute G = E * F
    for (i = 0; i < ni; i++) {
        for (j = 0; j < nl; j++) {
            double sum = 0.0;
            for (k = 0; k < nj; ++k) {
#pragma HLS PIPELINE II=1
                sum += E_local[i][k] * F_local[k][j];
            }
            G[i][j] = sum;
        }
    }
}

} // extern "C"