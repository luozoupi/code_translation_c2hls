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

  // Local buffers to stage data and enable reuse / parallel access
  double A_loc[NI][NK];
  double B_loc[NK][NJ];
  double C_loc[NJ][NL];
  double tmp_loc[NI][NJ];
  double D_loc[NI][NL];
#pragma HLS ARRAY_PARTITION variable=A_loc   cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=B_loc   cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=tmp_loc cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=C_loc   cyclic factor=8 dim=1

  // Load A
  LOAD_A_I:
  for (i = 0; i < ni; i++)
    LOAD_A_K:
    for (k = 0; k < nk; k++) {
#pragma HLS PIPELINE II=1
      A_loc[i][k] = A[i][k];
    }

  // Load B
  LOAD_B_K:
  for (k = 0; k < nk; k++)
    LOAD_B_J:
    for (j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
      B_loc[k][j] = B[k][j];
    }

  // Load C
  LOAD_C_K:
  for (k = 0; k < nj; k++)
    LOAD_C_J:
    for (j = 0; j < nl; j++) {
#pragma HLS PIPELINE II=1
      C_loc[k][j] = C[k][j];
    }

  // Load D
  LOAD_D_I:
  for (i = 0; i < ni; i++)
    LOAD_D_J:
    for (j = 0; j < nl; j++) {
#pragma HLS PIPELINE II=1
      D_loc[i][j] = D[i][j];
    }

  // First matrix multiply: tmp = alpha * A * B
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
      {
#pragma HLS PIPELINE II=1
	double acc = 0.0;
	for (k = 0; k < nk; ++k) {
#pragma HLS UNROLL
	  acc += alpha * A_loc[i][k] * B_loc[k][j];
	}
	tmp_loc[i][j] = acc;
      }

  // Second matrix multiply: D = beta * D + tmp * C
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++)
      {
#pragma HLS PIPELINE II=1
	double acc = D_loc[i][j] * beta;
	for (k = 0; k < nj; ++k) {
#pragma HLS UNROLL
	  acc += tmp_loc[i][k] * C_loc[k][j];
	}
	D_loc[i][j] = acc;
      }

  // Store tmp
  STORE_TMP_I:
  for (i = 0; i < ni; i++)
    STORE_TMP_J:
    for (j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
      tmp[i][j] = tmp_loc[i][j];
    }

  // Store D
  STORE_D_I:
  for (i = 0; i < ni; i++)
    STORE_D_J:
    for (j = 0; j < nl; j++) {
#pragma HLS PIPELINE II=1
      D[i][j] = D_loc[i][j];
    }
}