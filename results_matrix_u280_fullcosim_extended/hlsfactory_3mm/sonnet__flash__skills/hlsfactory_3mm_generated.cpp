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

    int i, j, k;

    // E = A * B  (ni x nj) = (ni x nk) * (nk x nj)
    for (i = 0; i < ni; i++)
        for (j = 0; j < nj; j++)
        {
            E[i][j] = 0.0;
            // Keep reduction serial to preserve FP bit-exact result
            for (k = 0; k < nk; ++k)
            {
#pragma HLS PIPELINE II=1
                E[i][j] += A[i][k] * B[k][j];
            }
        }

    // F = C * D  (nj x nl) = (nj x nm) * (nm x nl)
    for (i = 0; i < nj; i++)
        for (j = 0; j < nl; j++)
        {
            F[i][j] = 0.0;
            // Keep reduction serial to preserve FP bit-exact result
            for (k = 0; k < nm; ++k)
            {
#pragma HLS PIPELINE II=1
                F[i][j] += C[i][k] * D[k][j];
            }
        }

    // G = E * F  (ni x nl) = (ni x nj) * (nj x nl)
    for (i = 0; i < ni; i++)
        for (j = 0; j < nl; j++)
        {
            G[i][j] = 0.0;
            // Keep reduction serial to preserve FP bit-exact result
            for (k = 0; k < nj; ++k)
            {
#pragma HLS PIPELINE II=1
                G[i][j] += E[i][k] * F[k][j];
            }
        }
}

} // extern "C"