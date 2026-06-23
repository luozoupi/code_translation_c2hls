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
#pragma HLS INTERFACE m_axi port=A  offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=u1 offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=v1 offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=u2 offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=v2 offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=w  offset=slave bundle=gmem5
#pragma HLS INTERFACE m_axi port=x  offset=slave bundle=gmem6
#pragma HLS INTERFACE m_axi port=y  offset=slave bundle=gmem7
#pragma HLS INTERFACE m_axi port=z  offset=slave bundle=gmem8

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

  // Local buffers to enable efficient on-chip computation and partitioning
  static double A_loc[N][N];
#pragma HLS ARRAY_PARTITION variable=A_loc cyclic factor=8 dim=2

  static double u1_loc[N], v1_loc[N], u2_loc[N], v2_loc[N];
  static double w_loc[N], x_loc[N], y_loc[N], z_loc[N];
#pragma HLS ARRAY_PARTITION variable=v1_loc cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=v2_loc cyclic factor=8 dim=1

  // Load vectors
  load_vecs:
  for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
    u1_loc[i] = u1[i];
    v1_loc[i] = v1[i];
    u2_loc[i] = u2[i];
    v2_loc[i] = v2[i];
    w_loc[i]  = w[i];
    x_loc[i]  = x[i];
    y_loc[i]  = y[i];
    z_loc[i]  = z[i];
  }

  // Load A
  load_A_i:
  for (i = 0; i < n; i++) {
    load_A_j:
    for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
      A_loc[i][j] = A[i][j];
    }
  }

  // A = A + u1*v1 + u2*v2
  comp1_i:
  for (i = 0; i < n; i++) {
    double u1i = u1_loc[i];
    double u2i = u2_loc[i];
    comp1_j:
    for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
      A_loc[i][j] = A_loc[i][j] + u1i * v1_loc[j] + u2i * v2_loc[j];
    }
  }

  // x = x + beta * A^T * y
  comp2_i:
  for (i = 0; i < n; i++) {
    double acc = x_loc[i];
    comp2_j:
    for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
      acc = acc + beta * A_loc[j][i] * y_loc[j];
    }
    x_loc[i] = acc;
  }

  // x = x + z
  comp3:
  for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
    x_loc[i] = x_loc[i] + z_loc[i];
  }

  // w = w + alpha * A * x
  comp4_i:
  for (i = 0; i < n; i++) {
    double acc = w_loc[i];
    comp4_j:
    for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
      acc = acc + alpha * A_loc[i][j] * x_loc[j];
    }
    w_loc[i] = acc;
  }

  // Store results back
  store_A_i:
  for (i = 0; i < n; i++) {
    store_A_j:
    for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
      A[i][j] = A_loc[i][j];
    }
  }

  store_vecs:
  for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
    w[i] = w_loc[i];
    x[i] = x_loc[i];
  }
}
}