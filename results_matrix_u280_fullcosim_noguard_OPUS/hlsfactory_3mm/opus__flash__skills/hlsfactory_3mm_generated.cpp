#include "3mm.h"


extern "C" {
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

    // Local buffers to enable reuse and parallel access
    double l_A[NI][NK];
    double l_B[NK][NJ];
    double l_E[NI][NJ];
    double l_C[NJ][NM];
    double l_D[NM][NL];
    double l_F[NJ][NL];
    double l_G[NI][NL];
#pragma HLS ARRAY_PARTITION variable=l_A cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_B cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_C cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_D cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_E cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_F cyclic factor=8 dim=1

    // Load A
    LOAD_A_I: for (i = 0; i < ni; i++)
      LOAD_A_K: for (k = 0; k < nk; k++) {
#pragma HLS PIPELINE II=1
        l_A[i][k] = A[i][k];
      }

    // Load B
    LOAD_B_K: for (k = 0; k < nk; k++)
      LOAD_B_J: for (j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
        l_B[k][j] = B[k][j];
      }

    // Load C
    LOAD_C_I: for (i = 0; i < nj; i++)
      LOAD_C_K: for (k = 0; k < nm; k++) {
#pragma HLS PIPELINE II=1
        l_C[i][k] = C[i][k];
      }

    // Load D
    LOAD_D_K: for (k = 0; k < nm; k++)
      LOAD_D_J: for (j = 0; j < nl; j++) {
#pragma HLS PIPELINE II=1
        l_D[k][j] = D[k][j];
      }

  // E := A*B
  E_I: for (i = 0; i < ni; i++)
    E_J: for (j = 0; j < nj; j++)
      {
#pragma HLS PIPELINE II=1
	double acc = 0.0;
	E_K: for (k = 0; k < nk; ++k) {
#pragma HLS UNROLL factor=8
	  acc += l_A[i][k] * l_B[k][j];
        }
	l_E[i][j] = acc;
      }

  // F := C*D
  F_I: for (i = 0; i < nj; i++)
    F_J: for (j = 0; j < nl; j++)
      {
#pragma HLS PIPELINE II=1
	double acc = 0.0;
	F_K: for (k = 0; k < nm; ++k) {
#pragma HLS UNROLL factor=8
	  acc += l_C[i][k] * l_D[k][j];
        }
	l_F[i][j] = acc;
      }

  // G := E*F
  G_I: for (i = 0; i < ni; i++)
    G_J: for (j = 0; j < nl; j++)
      {
#pragma HLS PIPELINE II=1
	double acc = 0.0;
	G_K: for (k = 0; k < nj; ++k) {
#pragma HLS UNROLL factor=8
	  acc += l_E[i][k] * l_F[k][j];
        }
	l_G[i][j] = acc;
      }

    // Store E
    STORE_E_I: for (i = 0; i < ni; i++)
      STORE_E_J: for (j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
        E[i][j] = l_E[i][j];
      }

    // Store F
    STORE_F_I: for (i = 0; i < nj; i++)
      STORE_F_J: for (j = 0; j < nl; j++) {
#pragma HLS PIPELINE II=1
        F[i][j] = l_F[i][j];
      }

    // Store G
    STORE_G_I: for (i = 0; i < ni; i++)
      STORE_G_J: for (j = 0; j < nl; j++) {
#pragma HLS PIPELINE II=1
        G[i][j] = l_G[i][j];
      }

}
}