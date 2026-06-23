#include "3mm.h"


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

  // Local staging buffers to enable reuse and parallel access
  static double l_A[NI][NK];
  static double l_B[NK][NJ];
  static double l_E[NI][NJ];
  static double l_C[NJ][NM];
  static double l_D[NM][NL];
  static double l_F[NJ][NL];
  static double l_G[NI][NL];
#pragma HLS ARRAY_PARTITION variable=l_A cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_B cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_C cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_D cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_E cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_F cyclic factor=8 dim=1

  // Load A
  for (i = 0; i < ni; i++)
    for (k = 0; k < nk; k++) {
#pragma HLS PIPELINE II=1
      l_A[i][k] = A[i][k];
    }

  // Load B
  for (k = 0; k < nk; k++)
    for (j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
      l_B[k][j] = B[k][j];
    }

  // Load C
  for (i = 0; i < nj; i++)
    for (k = 0; k < nm; k++) {
#pragma HLS PIPELINE II=1
      l_C[i][k] = C[i][k];
    }

  // Load D
  for (k = 0; k < nm; k++)
    for (j = 0; j < nl; j++) {
#pragma HLS PIPELINE II=1
      l_D[k][j] = D[k][j];
    }

  // E = A * B
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
      {
#pragma HLS PIPELINE II=1
	double acc = 0.0;
	for (k = 0; k < nk; ++k) {
#pragma HLS UNROLL
	  acc += l_A[i][k] * l_B[k][j];
	}
	l_E[i][j] = acc;
      }

  // F = C * D
  for (i = 0; i < nj; i++)
    for (j = 0; j < nl; j++)
      {
#pragma HLS PIPELINE II=1
	double acc = 0.0;
	for (k = 0; k < nm; ++k) {
#pragma HLS UNROLL
	  acc += l_C[i][k] * l_D[k][j];
	}
	l_F[i][j] = acc;
      }

  // G = E * F
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++)
      {
#pragma HLS PIPELINE II=1
	double acc = 0.0;
	for (k = 0; k < nj; ++k) {
#pragma HLS UNROLL
	  acc += l_E[i][k] * l_F[k][j];
	}
	l_G[i][j] = acc;
      }

  // Store E
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
      E[i][j] = l_E[i][j];
    }

  // Store F
  for (i = 0; i < nj; i++)
    for (j = 0; j < nl; j++) {
#pragma HLS PIPELINE II=1
      F[i][j] = l_F[i][j];
    }

  // Store G
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
#pragma HLS PIPELINE II=1
      G[i][j] = l_G[i][j];
    }
}