#include "doitgen.h"

void kernel_doitgen(  
		    double A[ NR + 0][NQ + 0][NP + 0],
		    double C4[ NP + 0][NP + 0],
		    double sum[ NP + 0])
{
#pragma HLS INTERFACE m_axi port=A   offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=C4  offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=sum offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=A      bundle=control
#pragma HLS INTERFACE s_axilite port=C4     bundle=control
#pragma HLS INTERFACE s_axilite port=sum    bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int nr = NR;
    const int nq = NQ;
    const int np = NP;

    // Local buffers to avoid repeated global memory accesses
    double l_A[NR][NQ][NP];
    double l_C4[NP][NP];
    double l_sum[NP];

#pragma HLS ARRAY_PARTITION variable=l_A   cyclic factor=8 dim=3
#pragma HLS ARRAY_PARTITION variable=l_C4  cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_C4  cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_sum complete dim=1

    // Load A into local buffer
    for (int r = 0; r < NR; r++)
        for (int q = 0; q < NQ; q++)
            for (int p = 0; p < NP; p++) {
#pragma HLS PIPELINE II=1
                l_A[r][q][p] = A[r][q][p];
            }

    // Load C4 into local buffer
    for (int s = 0; s < NP; s++)
        for (int p = 0; p < NP; p++) {
#pragma HLS PIPELINE II=1
            l_C4[s][p] = C4[s][p];
        }

    int r, q, p, s;

    for (r = 0; r < nr; r++)
        for (q = 0; q < nq; q++) {
            for (p = 0; p < np; p++) {
#pragma HLS PIPELINE II=1
                double acc = 0.0;
                for (s = 0; s < np; s++) {
#pragma HLS UNROLL factor=8
                    acc += l_A[r][q][s] * l_C4[s][p];
                }
                l_sum[p] = acc;
            }
            for (p = 0; p < np; p++) {
#pragma HLS PIPELINE II=1
                l_A[r][q][p] = l_sum[p];
            }
        }

    // Write A back to global memory
    for (int r = 0; r < NR; r++)
        for (int q = 0; q < NQ; q++)
            for (int p = 0; p < NP; p++) {
#pragma HLS PIPELINE II=1
                A[r][q][p] = l_A[r][q][p];
            }

    // Write sum back to global memory
    for (int p = 0; p < NP; p++) {
#pragma HLS PIPELINE II=1
        sum[p] = l_sum[p];
    }
}