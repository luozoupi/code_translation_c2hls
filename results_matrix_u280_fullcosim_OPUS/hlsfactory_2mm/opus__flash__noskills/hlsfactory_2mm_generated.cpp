#include "2mm.h"

extern "C" {
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


  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
      {
#pragma HLS PIPELINE II=1
	tmp[i][j] = 0.0;
	double acc0 = 0.0;
	for (k = 0; k < nk; ++k)
	  acc0 += alpha * A[i][k] * B[k][j];
	tmp[i][j] = acc0;
      }
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++)
      {
#pragma HLS PIPELINE II=1
	double acc1 = D[i][j] * beta;
	for (k = 0; k < nj; ++k)
	  acc1 += tmp[i][k] * C[k][j];
	D[i][j] = acc1;
      }

}
}