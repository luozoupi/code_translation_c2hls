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

  // Local working buffers staged from global memory for reuse across phases
  static double data_l[N][M];
#pragma HLS ARRAY_PARTITION variable=data_l cyclic factor=8 dim=2
  static double mean_l[M];
  static double stddev_l[M];

  // Stage data into local memory
  LOAD_I: for (i = 0; i < n; i++)
    LOAD_J: for (j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
      data_l[i][j] = data[i][j];
    }

  // Mean computation
  MEAN_J: for (j = 0; j < m; j++)
    {
      double acc = 0.0;
      MEAN_I: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        acc += data_l[i][j];
      }
      mean_l[j] = acc / float_n;
    }

  // Standard deviation computation
  STD_J: for (j = 0; j < m; j++)
    {
      double acc = 0.0;
      double mj = mean_l[j];
      STD_I: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        double diff = data_l[i][j] - mj;
        acc += diff * diff;
      }
      acc /= float_n;
      acc = sqrt(acc);
      stddev_l[j] = acc <= eps ? 1.0 : acc;
    }

  // Center and normalize the data matrix
  double sqrt_fn = sqrt(float_n);
  NORM_I: for (i = 0; i < n; i++)
    NORM_J: for (j = 0; j < m; j++)
      {
#pragma HLS PIPELINE II=1
        double v = data_l[i][j] - mean_l[j];
        v /= sqrt_fn * stddev_l[j];
        data_l[i][j] = v;
      }

  // Correlation matrix
  CORR_I: for (i = 0; i < m-1; i++)
    {
      corr[i][i] = 1.0;
      CORR_J: for (j = i+1; j < m; j++)
        {
          double acc = 0.0;
          CORR_K: for (k = 0; k < n; k++) {
#pragma HLS PIPELINE II=1
            acc += (data_l[k][i] * data_l[k][j]);
          }
          corr[i][j] = acc;
          corr[j][i] = acc;
        }
    }
  corr[m-1][m-1] = 1.0;

  // Write back centered/normalized data and statistics
  STORE_I: for (i = 0; i < n; i++)
    STORE_J: for (j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
      data[i][j] = data_l[i][j];
    }

  STORE_STATS: for (j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
    mean[j]   = mean_l[j];
    stddev[j] = stddev_l[j];
  }
}