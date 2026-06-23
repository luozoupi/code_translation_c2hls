#include "gemver.h"


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

  // Stage scalar/vector working sets locally for reuse across the loop nests.
  double l_u1[N], l_v1[N], l_u2[N], l_v2[N];
  double l_x[N], l_y[N], l_z[N], l_w[N];
  double l_A[N][N];

#pragma HLS ARRAY_PARTITION variable=l_v1 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_v2 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_x  cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_y  cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_A  cyclic factor=8 dim=2

  // ---- Load vectors ----
LOAD_VEC:
  for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
    l_u1[i] = u1[i];
    l_v1[i] = v1[i];
    l_u2[i] = u2[i];
    l_v2[i] = v2[i];
    l_x[i]  = x[i];
    l_y[i]  = y[i];
    l_z[i]  = z[i];
    l_w[i]  = w[i];
  }

  // ---- Load A ----
LOAD_A_I:
  for (i = 0; i < n; i++)
LOAD_A_J:
    for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
      l_A[i][j] = A[i][j];
    }

  // ---- A = A + u1*v1 + u2*v2 ----
K1_I:
  for (i = 0; i < n; i++) {
    double tu1 = l_u1[i];
    double tu2 = l_u2[i];
K1_J:
    for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
      l_A[i][j] = l_A[i][j] + tu1 * l_v1[j] + tu2 * l_v2[j];
    }
  }

  // ---- x = x + beta * A^T * y ----
K2_I:
  for (i = 0; i < n; i++) {
    double acc = l_x[i];
K2_J:
    for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
      acc = acc + beta * l_A[j][i] * l_y[j];
    }
    l_x[i] = acc;
  }

  // ---- x = x + z ----
K3:
  for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
    l_x[i] = l_x[i] + l_z[i];
  }

  // ---- w = w + alpha * A * x ----
K4_I:
  for (i = 0; i < n; i++) {
    double acc = l_w[i];
K4_J:
    for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
      acc = acc + alpha * l_A[i][j] * l_x[j];
    }
    l_w[i] = acc;
  }

  // ---- Store results back ----
STORE_A_I:
  for (i = 0; i < n; i++)
STORE_A_J:
    for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
      A[i][j] = l_A[i][j];
    }

STORE_VEC:
  for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
    x[i] = l_x[i];
    w[i] = l_w[i];
  }

}