#include "gemm.h"


void kernel_gemm(  
		 double alpha,
		 double beta,
		 double C[ NI + 0][NJ + 0],
		 double A[ NI + 0][NK + 0],
		 double B[ NK + 0][NJ + 0])
{
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=alpha bundle=control
#pragma HLS INTERFACE s_axilite port=beta bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int ni = NI;
    const int nj = NJ;
    const int nk = NK;

  int i, j, k;

  // Stage the full B matrix into local BRAM once (reused across all rows of C).
  static double B_local[NK][NJ];
  for (k = 0; k < nk; k++) {
    for (j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
      B_local[k][j] = B[k][j];
    }
  }

  for (i = 0; i < ni; i++) {
    // Local row buffers to eliminate per-iteration AXI traffic in the hot loop.
    double C_row[NJ];
    double A_row[NK];

    // Load row i of A into local memory.
    for (k = 0; k < nk; k++) {
#pragma HLS PIPELINE II=1
      A_row[k] = A[i][k];
    }

    // Load and scale row i of C.
    for (j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
      C_row[j] = C[i][j] * beta;
    }

    // Compute. Accumulation order over k is preserved exactly (serial),
    // so the bit-exact result matches the original kernel.
    for (k = 0; k < nk; k++) {
      double a_val = alpha * A_row[k];
      for (j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
        // Inner loop touches distinct C_row[j] elements each iteration,
        // so no same-address loop-carried dependence within this loop.
        C_row[j] += a_val * B_local[k][j];
      }
    }

    // Write row i of C back to global memory.
    for (j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
      C[i][j] = C_row[j];
    }
  }

}