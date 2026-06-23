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
#pragma HLS INTERFACE s_axilite port=A   bundle=control
#pragma HLS INTERFACE s_axilite port=C4  bundle=control
#pragma HLS INTERFACE s_axilite port=sum bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int nr = NR;
    const int nq = NQ;
    const int np = NP;

  int r, q, p, s;

  // Local staging buffers for reuse within each (r,q) iteration.
  double C4_local[NP][NP];
#pragma HLS ARRAY_PARTITION variable=C4_local complete dim=1

  double a_local[NP];
#pragma HLS ARRAY_PARTITION variable=a_local complete dim=1

  double sum_local[NP];
#pragma HLS ARRAY_PARTITION variable=sum_local complete dim=1

  // Stage C4 once: it is reused across all (r,q).
  for (s = 0; s < np; s++) {
    for (p = 0; p < np; p++) {
#pragma HLS PIPELINE II=1
      C4_local[s][p] = C4[s][p];
    }
  }

  for (r = 0; r < nr; r++) {
    for (q = 0; q < nq; q++)  {

      // Load A[r][q][*] into local buffer for reuse in reduction.
      for (s = 0; s < np; s++) {
#pragma HLS PIPELINE II=1
        a_local[s] = A[r][q][s];
      }

      // Compute sum[p] = sum_s A[r][q][s] * C4[s][p]
      for (p = 0; p < np; p++)  {
#pragma HLS PIPELINE II=1
        double acc = 0.0;
        for (s = 0; s < np; s++) {
#pragma HLS UNROLL
          acc += a_local[s] * C4_local[s][p];
        }
        sum_local[p] = acc;
      }

      // Write back results into A and external sum.
      for (p = 0; p < np; p++) {
#pragma HLS PIPELINE II=1
        A[r][q][p] = sum_local[p];
        sum[p] = sum_local[p];
      }
    }
  }

}
}