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

    // Local buffers for reduced global-memory traffic
    double l_A[NQ][NP];
    double l_C4[NP][NP];
    double l_sum[NP];

#pragma HLS ARRAY_PARTITION variable=l_A   cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_C4  cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_sum cyclic factor=8 dim=1

    // Pre-load C4 once into local buffer
    load_c4_row: for (int s = 0; s < NP; s++) {
        load_c4_col: for (int p = 0; p < NP; p++) {
#pragma HLS PIPELINE II=1
            l_C4[s][p] = C4[s][p];
        }
    }

    int r, q, p, s;

    for (r = 0; r < nr; r++) {

        // Load A[r][*][*] into local buffer
        load_a_q: for (q = 0; q < nq; q++) {
            load_a_p: for (p = 0; p < np; p++) {
#pragma HLS PIPELINE II=1
                l_A[q][p] = A[r][q][p];
            }
        }

        // Compute
        for (q = 0; q < nq; q++) {

            // Initialize sum and compute dot products
            for (p = 0; p < np; p++) {
                l_sum[p] = 0.0;
                for (s = 0; s < np; s++) {
#pragma HLS PIPELINE II=1
                    l_sum[p] += l_A[q][s] * l_C4[s][p];
                }
            }

            // Write back sum into local A tile
            write_back: for (p = 0; p < np; p++) {
#pragma HLS PIPELINE II=1
                l_A[q][p] = l_sum[p];
            }
        }

        // Store result back to global A
        store_a_q: for (q = 0; q < nq; q++) {
            store_a_p: for (p = 0; p < np; p++) {
#pragma HLS PIPELINE II=1
                A[r][q][p] = l_A[q][p];
            }
        }
    }

    // Write final sum[] output (last r,q slice sum values)
    // Preserve original interface: write l_sum back to global sum
    write_sum: for (p = 0; p < np; p++) {
#pragma HLS PIPELINE II=1
        sum[p] = l_sum[p];
    }
}