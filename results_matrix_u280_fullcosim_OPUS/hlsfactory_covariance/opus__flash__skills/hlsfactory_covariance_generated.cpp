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

  // Local buffers for reuse and partitioned parallel access
  static double l_mean[M];
  static double l_data[N][M];
#pragma HLS ARRAY_PARTITION variable=l_mean complete dim=1
#pragma HLS ARRAY_PARTITION variable=l_data cyclic factor=8 dim=2

  // Stage input data into local buffer
  LOAD_I: for (i = 0; i < n; i++)
    LOAD_J: for (j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
      l_data[i][j] = data[i][j];
    }

  // Compute mean per column
  MEAN_J: for (j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
    double acc = 0.0;
    MEAN_I: for (i = 0; i < n; i++) {
      acc += l_data[i][j];
    }
    l_mean[j] = acc / float_n;
  }

  // Subtract mean
  SUB_I: for (i = 0; i < n; i++)
    SUB_J: for (j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
      l_data[i][j] -= l_mean[j];
    }

  // Covariance computation
  COV_I: for (i = 0; i < m; i++)
    COV_J: for (j = i; j < m; j++)
      {
        double acc = 0.0;
        COV_K: for (k = 0; k < n; k++) {
#pragma HLS PIPELINE II=1
	  acc += l_data[k][i] * l_data[k][j];
        }
        double val = acc / (float_n - 1.0);
        cov[i][j] = val;
        cov[j][i] = val;
      }

  // Write mean back
  STORE_MEAN: for (j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
    mean[j] = l_mean[j];
  }

  // Write modified data back
  STORE_I: for (i = 0; i < n; i++)
    STORE_J: for (j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
      data[i][j] = l_data[i][j];
    }
}
}