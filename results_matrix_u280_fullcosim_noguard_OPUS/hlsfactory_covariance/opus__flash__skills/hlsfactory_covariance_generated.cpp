#include "covariance.h"

extern "C" {
void kernel_covariance(
		       double float_n,
		       double data[ N + 0][M + 0],
		       double cov[ M + 0][M + 0],
		       double mean[ M + 0])
{
#pragma HLS INTERFACE m_axi port=data offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=cov  offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=mean offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=float_n bundle=control
#pragma HLS INTERFACE s_axilite port=data    bundle=control
#pragma HLS INTERFACE s_axilite port=cov     bundle=control
#pragma HLS INTERFACE s_axilite port=mean    bundle=control
#pragma HLS INTERFACE s_axilite port=return  bundle=control

    const int n = N;
    const int m = M;

  int i, j, k;

  // Local staging buffers for reuse across the multiple passes.
  static double data_local[N][M];
#pragma HLS ARRAY_PARTITION variable=data_local cyclic factor=8 dim=2
  static double mean_local[M];
#pragma HLS ARRAY_PARTITION variable=mean_local cyclic factor=8 dim=1

  // Stage input data into local memory.
  load_data_i:
  for (i = 0; i < n; i++) {
    load_data_j:
    for (j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
      data_local[i][j] = data[i][j];
    }
  }

  // Compute means.
  mean_j:
  for (j = 0; j < m; j++) {
    double acc = 0.0;
    mean_i:
    for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
      acc += data_local[i][j];
    }
    mean_local[j] = acc / float_n;
    mean[j] = mean_local[j];
  }

  // Center the data.
  center_i:
  for (i = 0; i < n; i++) {
    center_j:
    for (j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
      data_local[i][j] -= mean_local[j];
    }
  }

  // Covariance computation.
  cov_i:
  for (i = 0; i < m; i++) {
    cov_j:
    for (j = i; j < m; j++) {
      double acc = 0.0;
      cov_k:
      for (k = 0; k < n; k++) {
#pragma HLS PIPELINE II=1
        acc += data_local[k][i] * data_local[k][j];
      }
      acc /= (float_n - 1.0);
      cov[i][j] = acc;
      cov[j][i] = acc;
    }
  }

}
}