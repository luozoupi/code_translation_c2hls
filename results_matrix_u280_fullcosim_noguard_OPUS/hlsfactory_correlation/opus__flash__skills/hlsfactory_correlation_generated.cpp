#include "correlation.h"

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

  // Local buffers to stage global memory for reuse
  static double l_data[N][M];
#pragma HLS ARRAY_PARTITION variable=l_data cyclic factor=8 dim=2
  static double l_mean[M];
#pragma HLS ARRAY_PARTITION variable=l_mean cyclic factor=8 dim=1
  static double l_stddev[M];
#pragma HLS ARRAY_PARTITION variable=l_stddev cyclic factor=8 dim=1

  // Stage data into local memory
  LOAD_I: for (i = 0; i < n; i++) {
    LOAD_J: for (j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
      l_data[i][j] = data[i][j];
    }
  }

  // Compute mean
  MEAN_J: for (j = 0; j < m; j++)
    {
      double acc = 0.0;
      MEAN_I: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
	acc += l_data[i][j];
      }
      l_mean[j] = acc / float_n;
    }

  // Compute stddev
  STDDEV_J: for (j = 0; j < m; j++)
    {
      double acc = 0.0;
      double mj = l_mean[j];
      STDDEV_I: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        double diff = l_data[i][j] - mj;
        acc += diff * diff;
      }
      acc /= float_n;
      acc = sqrt(acc);
      l_stddev[j] = acc <= eps ? 1.0 : acc;
    }

  // Normalize data
  double sqrt_fn = sqrt(float_n);
  NORM_I: for (i = 0; i < n; i++)
    NORM_J: for (j = 0; j < m; j++)
      {
#pragma HLS PIPELINE II=1
        double v = l_data[i][j] - l_mean[j];
        v /= sqrt_fn * l_stddev[j];
        l_data[i][j] = v;
      }

  // Compute correlation matrix
  CORR_I: for (i = 0; i < m-1; i++)
    {
      corr[i][i] = 1.0;
      CORR_J: for (j = i+1; j < m; j++)
        {
          double acc = 0.0;
          CORR_K: for (k = 0; k < n; k++) {
#pragma HLS PIPELINE II=1
            acc += (l_data[k][i] * l_data[k][j]);
          }
          corr[i][j] = acc;
          corr[j][i] = acc;
        }
    }
  corr[m-1][m-1] = 1.0;

  // Write back normalized data and statistics
  STORE_I: for (i = 0; i < n; i++) {
    STORE_J: for (j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
      data[i][j] = l_data[i][j];
    }
  }
  STORE_STATS: for (j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
    mean[j] = l_mean[j];
    stddev[j] = l_stddev[j];
  }
}