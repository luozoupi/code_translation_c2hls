#include "gemver.h"


extern "C" {
void kernel_gemver(
		   double alpha,
		   double beta,
		   double A[ N + 0][N + 0],
		   double u1[ N + 0],
		   double v1[ N + 0],
		   double u2[ N + 0],
		   double v2[ N + 0],
		   double w[ N + 0],
		   double x[ N + 0],
		   double y[ N + 0],
		   double z[ N + 0])
{
#pragma HLS INTERFACE m_axi port=A   offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=u1  offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=v1  offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=u2  offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=v2  offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=w   offset=slave bundle=gmem5
#pragma HLS INTERFACE m_axi port=x   offset=slave bundle=gmem6
#pragma HLS INTERFACE m_axi port=y   offset=slave bundle=gmem7
#pragma HLS INTERFACE m_axi port=z   offset=slave bundle=gmem8

#pragma HLS INTERFACE s_axilite port=alpha bundle=control
#pragma HLS INTERFACE s_axilite port=beta  bundle=control
#pragma HLS INTERFACE s_axilite port=A     bundle=control
#pragma HLS INTERFACE s_axilite port=u1    bundle=control
#pragma HLS INTERFACE s_axilite port=v1    bundle=control
#pragma HLS INTERFACE s_axilite port=u2    bundle=control
#pragma HLS INTERFACE s_axilite port=v2    bundle=control
#pragma HLS INTERFACE s_axilite port=w     bundle=control
#pragma HLS INTERFACE s_axilite port=x     bundle=control
#pragma HLS INTERFACE s_axilite port=y     bundle=control
#pragma HLS INTERFACE s_axilite port=z     bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;

  int i, j;

  // ---- Stage 1: A = A + u1*v1' + u2*v2'  (fully independent per element) ----
  L1_i: for (i = 0; i < n; i++) {
    double u1i = u1[i];
    double u2i = u2[i];
    L1_j: for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
      A[i][j] = A[i][j] + u1i * v1[j] + u2i * v2[j];
    }
  }

  // ---- Stage 2: x[i] += beta * A[j][i] * y[j]  (serial FP reduction) ----
  // Keep accumulation order bit-exact: pipeline body, do not reassociate.
  L2_i: for (i = 0; i < n; i++) {
    double xi = x[i];
    L2_j: for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
      xi = xi + beta * A[j][i] * y[j];
    }
    x[i] = xi;
  }

  // ---- Stage 3: x[i] += z[i]  (independent per element) ----
  L3_i: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
    x[i] = x[i] + z[i];
  }

  // ---- Stage 4: w[i] += alpha * A[i][j] * x[j]  (serial FP reduction) ----
  L4_i: for (i = 0; i < n; i++) {
    double wi = w[i];
    L4_j: for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
      wi = wi + alpha * A[i][j] * x[j];
    }
    w[i] = wi;
  }

}
}