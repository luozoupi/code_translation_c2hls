#include "correlation.h"
#include <cstring>


void kernel_correlation( 
			double float_n,
			double data[ N + 0][M + 0],
			double corr[ M + 0][M + 0],
			double mean[ M + 0],
			double stddev[ M + 0])
{
#pragma HLS INTERFACE m_axi port=data   offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=corr   offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=mean   offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=stddev offset=slave bundle=gmem3
#pragma HLS INTERFACE s_axilite port=float_n bundle=control
#pragma HLS INTERFACE s_axilite port=data    bundle=control
#pragma HLS INTERFACE s_axilite port=corr    bundle=control
#pragma HLS INTERFACE s_axilite port=mean    bundle=control
#pragma HLS INTERFACE s_axilite port=stddev  bundle=control
#pragma HLS INTERFACE s_axilite port=return  bundle=control

    const int n = N;
    const int m = M;

  int i, j, k;

  double eps = 0.1;

  // ---- Local tile buffers ----
  double data_local[N][M];
  double mean_local[M];
  double stddev_local[M];
  double corr_local[M][M];

  // ---- LOAD phase: stage data into local buffer ----
  load_data:
  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
      data_local[i][j] = data[i][j];
    }
  }

  // ---- COMPUTE phase: mean ----
  compute_mean:
  for (j = 0; j < m; j++)
    {
      mean_local[j] = 0.0;
      for (i = 0; i < n; i++)
#pragma HLS PIPELINE II=1
	mean_local[j] += data_local[i][j];
      mean_local[j] /= float_n;
    }

  // ---- COMPUTE phase: stddev ----
  compute_stddev:
   for (j = 0; j < m; j++)
    {
      stddev_local[j] = 0.0;
      for (i = 0; i < n; i++)
#pragma HLS PIPELINE II=1
        stddev_local[j] += (data_local[i][j] - mean_local[j]) * (data_local[i][j] - mean_local[j]);
      stddev_local[j] /= float_n;
      stddev_local[j] = sqrt(stddev_local[j]);

      stddev_local[j] = stddev_local[j] <= eps ? 1.0 : stddev_local[j];
    }

  // ---- COMPUTE phase: center and normalize data ----
  compute_normalize:
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++)
      {
#pragma HLS PIPELINE II=1
        data_local[i][j] -= mean_local[j];
        data_local[i][j] /= sqrt(float_n) * stddev_local[j];
      }

  // ---- COMPUTE phase: correlation ----
  compute_corr:
  for (i = 0; i < m-1; i++)
    {
      corr_local[i][i] = 1.0;
      for (j = i+1; j < m; j++)
        {
          double acc = 0.0;
          for (k = 0; k < n; k++)
#pragma HLS PIPELINE II=1
            acc += (data_local[k][i] * data_local[k][j]);
          corr_local[i][j] = acc;
          corr_local[j][i] = acc;
        }
    }
  corr_local[m-1][m-1] = 1.0;

  // ---- STORE phase: write results back to global memory ----
  store_mean:
  for (j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
    mean[j] = mean_local[j];
    stddev[j] = stddev_local[j];
  }

  store_data:
  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
      data[i][j] = data_local[i][j];
    }
  }

  store_corr:
  for (i = 0; i < m; i++) {
    for (j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
      corr[i][j] = corr_local[i][j];
    }
  }

}