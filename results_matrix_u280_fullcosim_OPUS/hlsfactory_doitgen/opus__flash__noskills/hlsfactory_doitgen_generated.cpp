#include "doitgen.h"

extern "C" {
void kernel_doitgen(
		    double A[ NR + 0][NQ + 0][NP + 0],
		    double C4[ NP + 0][NP + 0],
		    double sum[ NP + 0])
{
#pragma HLS INTERFACE m_axi port=A  offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=C4 offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=sum offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=A  bundle=control
#pragma HLS INTERFACE s_axilite port=C4 bundle=control
#pragma HLS INTERFACE s_axilite port=sum bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int nr = NR;
    const int nq = NQ;
    const int np = NP;

  int r, q, p, s;

  // Local buffers for parallel access
  double C4_local[NP][NP];
#pragma HLS ARRAY_PARTITION variable=C4_local complete dim=1

  double sum_local[NP];
#pragma HLS ARRAY_PARTITION variable=sum_local complete dim=1

  double A_row[NP];
#pragma HLS ARRAY_PARTITION variable=A_row complete dim=1

  // Cache C4 into local memory once
  for (s = 0; s < np; s++) {
    for (p = 0; p < np; p++) {
#pragma HLS PIPELINE II=1
      C4_local[s][p] = C4[s][p];
    }
  }

  for (r = 0; r < nr; r++) {
    for (q = 0; q < nq; q++)  {

      // Load A[r][q][*] into local registers
      for (s = 0; s < np; s++) {
#pragma HLS PIPELINE II=1
        A_row[s] = A[r][q][s];
      }

      for (p = 0; p < np; p++)  {
#pragma HLS PIPELINE II=1
        double acc = 0.0;
        for (s = 0; s < np; s++) {
#pragma HLS UNROLL
          acc += A_row[s] * C4_local[s][p];
        }
        sum_local[p] = acc;
      }

      for (p = 0; p < np; p++) {
#pragma HLS PIPELINE II=1
        A[r][q][p] = sum_local[p];
        sum[p] = sum_local[p];
      }
    }
  }

}
}