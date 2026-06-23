#include "doitgen.h"

void kernel_doitgen(
		    double A[ NR + 0][NQ + 0][NP + 0],
		    double C4[ NP + 0][NP + 0],
		    double sum[ NP + 0])
{
#pragma HLS INLINE off

    const int nr = NR;
    const int nq = NQ;
    const int np = NP;

  int r, q, p, s;

  // Local buffers for reuse across the q-loop (C4) and inner reductions.
  static double C4_local[NP][NP];
#pragma HLS ARRAY_PARTITION variable=C4_local complete dim=1

  double a_row[NP];
#pragma HLS ARRAY_PARTITION variable=a_row complete dim=1

  double sum_local[NP];
#pragma HLS ARRAY_PARTITION variable=sum_local complete dim=1

  // Stage C4 into local memory once (reused for all r,q).
  for (s = 0; s < np; s++) {
    for (p = 0; p < np; p++) {
#pragma HLS PIPELINE II=1
      C4_local[s][p] = C4[s][p];
    }
  }

  for (r = 0; r < nr; r++) {
    for (q = 0; q < nq; q++)  {

      // Stage current A[r][q] row locally for fast reuse.
      for (s = 0; s < np; s++) {
#pragma HLS PIPELINE II=1
        a_row[s] = A[r][q][s];
      }

      // Initialize partial sums.
      for (p = 0; p < np; p++) {
#pragma HLS UNROLL
        sum_local[p] = 0.0;
      }

      // Accumulate: for each s, update all p in parallel.
      for (s = 0; s < np; s++) {
#pragma HLS PIPELINE II=1
        double a_s = a_row[s];
        for (p = 0; p < np; p++) {
#pragma HLS UNROLL
          sum_local[p] += a_s * C4_local[s][p];
        }
      }

      // Write back results.
      for (p = 0; p < np; p++) {
#pragma HLS PIPELINE II=1
        sum[p] = sum_local[p];
        A[r][q][p] = sum_local[p];
      }
    }
  }
}

extern "C" {
void workload(
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

  kernel_doitgen(A, C4, sum);
}
}