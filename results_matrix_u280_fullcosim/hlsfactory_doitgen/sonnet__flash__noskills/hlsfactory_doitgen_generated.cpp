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

    // Local copies to avoid repeated global-memory traffic
    double local_A[NR][NQ][NP];
    double local_C4[NP][NP];
    double local_sum[NP];

#pragma HLS ARRAY_PARTITION variable=local_A   cyclic factor=4 dim=3
#pragma HLS ARRAY_PARTITION variable=local_C4  cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=local_sum complete

    // ---- Load A from global memory ----
    for (int r = 0; r < NR; r++)
        for (int q = 0; q < NQ; q++)
            for (int p = 0; p < NP; p++) {
#pragma HLS PIPELINE II=1
                local_A[r][q][p] = A[r][q][p];
            }

    // ---- Load C4 from global memory ----
    for (int s = 0; s < NP; s++)
        for (int p = 0; p < NP; p++) {
#pragma HLS PIPELINE II=1
            local_C4[s][p] = C4[s][p];
        }

    const int nr = NR;
    const int nq = NQ;
    const int np = NP;

    int r, q, p, s;

    for (r = 0; r < nr; r++)
        for (q = 0; q < nq; q++) {
            // Compute sum[p]
            for (p = 0; p < np; p++) {
#pragma HLS PIPELINE II=1
                double acc = 0.0;
                for (s = 0; s < np; s++) {
#pragma HLS UNROLL factor=4
                    acc += local_A[r][q][s] * local_C4[s][p];
                }
                local_sum[p] = acc;
            }
            // Write back to local_A
            for (p = 0; p < np; p++) {
#pragma HLS PIPELINE II=1
                local_A[r][q][p] = local_sum[p];
            }
        }

    // ---- Store A back to global memory ----
    for (int r = 0; r < NR; r++)
        for (int q = 0; q < NQ; q++)
            for (int p = 0; p < NP; p++) {
#pragma HLS PIPELINE II=1
                A[r][q][p] = local_A[r][q][p];
            }

    // ---- Store sum back to global memory ----
    for (int p = 0; p < NP; p++) {
#pragma HLS PIPELINE II=1
        sum[p] = local_sum[p];
    }
}

} // extern "C"