#include "doitgen.h"

extern "C" {

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

    // Local buffers to avoid repeated global memory traffic
    double local_A[NR][NQ][NP];
    double local_C4[NP][NP];
    double local_sum[NP];

#pragma HLS ARRAY_PARTITION variable=local_sum complete dim=1
#pragma HLS ARRAY_PARTITION variable=local_A   cyclic factor=2 dim=3
#pragma HLS ARRAY_PARTITION variable=local_C4  cyclic factor=2 dim=2

    // Load A from global memory
    for (int r = 0; r < nr; r++)
        for (int q = 0; q < nq; q++)
            for (int p = 0; p < np; p++) {
#pragma HLS PIPELINE II=1
                local_A[r][q][p] = A[r][q][p];
            }

    // Load C4 from global memory
    for (int s = 0; s < np; s++)
        for (int p = 0; p < np; p++) {
#pragma HLS PIPELINE II=1
            local_C4[s][p] = C4[s][p];
        }

    int r, q, p, s;

    for (r = 0; r < nr; r++)
        for (q = 0; q < nq; q++) {
            // Compute sum[p] for all p
            for (p = 0; p < np; p++) {
#pragma HLS PIPELINE II=1
                local_sum[p] = 0.0;
            }
            // Serial FP reduction over s — do NOT unroll s loop (bit-exact preservation)
            for (s = 0; s < np; s++) {
                for (p = 0; p < np; p++) {
#pragma HLS PIPELINE II=1
                    local_sum[p] += local_A[r][q][s] * local_C4[s][p];
                }
            }
            // Write sum back to A
            for (p = 0; p < np; p++) {
#pragma HLS PIPELINE II=1
                local_A[r][q][p] = local_sum[p];
            }
        }

    // Store A back to global memory
    for (int r2 = 0; r2 < nr; r2++)
        for (int q2 = 0; q2 < nq; q2++)
            for (int p2 = 0; p2 < np; p2++) {
#pragma HLS PIPELINE II=1
                A[r2][q2][p2] = local_A[r2][q2][p2];
            }

    // Write final sum to global memory (last r,q iteration)
    for (int p2 = 0; p2 < np; p2++) {
#pragma HLS PIPELINE II=1
        sum[p2] = local_sum[p2];
    }
}

} // extern "C"