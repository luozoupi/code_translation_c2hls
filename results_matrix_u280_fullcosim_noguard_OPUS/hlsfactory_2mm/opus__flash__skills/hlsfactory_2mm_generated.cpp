#include "2mm.h"


void kernel_2mm(   
		double alpha,
		double beta,
		double tmp[ NI + 0][NJ + 0],
		double A[ NI + 0][NK + 0],
		double B[ NK + 0][NJ + 0],
		double C[ NJ + 0][NL + 0],
		double D[ NI + 0][NL + 0])
{
#pragma HLS INTERFACE m_axi port=tmp offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=A   offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=B   offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=C   offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=D   offset=slave bundle=gmem4
#pragma HLS INTERFACE s_axilite port=alpha bundle=control
#pragma HLS INTERFACE s_axilite port=beta  bundle=control
#pragma HLS INTERFACE s_axilite port=tmp   bundle=control
#pragma HLS INTERFACE s_axilite port=A     bundle=control
#pragma HLS INTERFACE s_axilite port=B     bundle=control
#pragma HLS INTERFACE s_axilite port=C     bundle=control
#pragma HLS INTERFACE s_axilite port=D     bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int ni = NI;
    const int nj = NJ;
    const int nk = NK;
    const int nl = NL;

  int i, j, k;

  // ---- First matrix product: tmp = alpha * A * B ----
  // Stage one row of A locally for reuse across all columns j.
  for (i = 0; i < ni; i++)
  {
    double A_row[NK];
#pragma HLS ARRAY_PARTITION variable=A_row complete dim=1
    for (k = 0; k < nk; ++k) {
#pragma HLS PIPELINE II=1
      A_row[k] = A[i][k];
    }

    for (j = 0; j < nj; j++)
    {
#pragma HLS PIPELINE II=1
      double acc = 0.0;
      for (k = 0; k < nk; ++k) {
#pragma HLS UNROLL
        acc += alpha * A_row[k] * B[k][j];
      }
      tmp[i][j] = acc;
    }
  }

  // ---- Second matrix product: D = beta * D + tmp * C ----
  for (i = 0; i < ni; i++)
  {
    double tmp_row[NJ];
#pragma HLS ARRAY_PARTITION variable=tmp_row complete dim=1
    for (k = 0; k < nj; ++k) {
#pragma HLS PIPELINE II=1
      tmp_row[k] = tmp[i][k];
    }

    for (j = 0; j < nl; j++)
    {
#pragma HLS PIPELINE II=1
      double acc = D[i][j] * beta;
      for (k = 0; k < nj; ++k) {
#pragma HLS UNROLL
        acc += tmp_row[k] * C[k][j];
      }
      D[i][j] = acc;
    }
  }
}