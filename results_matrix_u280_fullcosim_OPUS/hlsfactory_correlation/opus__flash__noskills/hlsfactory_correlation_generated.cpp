#include "correlation.h"

extern "C" {
void kernel_correlation(
			double float_n,
			double data[ N + 0][M + 0],
			double corr[ M + 0][M + 0],
			double mean[ M + 0],
			double stddev[ M + 0])
{
#pragma HLS INTERFACE m_axi port=data offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=corr offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=mean offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=stddev offset=slave bundle=gmem3
#pragma HLS INTERFACE s_axilite port=float_n bundle=control
#pragma HLS INTERFACE s_axilite port=data bundle=control
#pragma HLS INTERFACE s_axilite port=corr bundle=control
#pragma HLS INTERFACE s_axilite port=mean bundle=control
#pragma HLS INTERFACE s_axilite port=stddev bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;
    const int m = M;

  int i, j, k;

  double eps = 0.1;


  mean_loop_j: for (j = 0; j < m; j++)
    {
      mean[j] = 0.0;
      mean_loop_i: for (i = 0; i < n; i++)
      {
#pragma HLS PIPELINE II=1
	mean[j] += data[i][j];
      }
      mean[j] /= float_n;
    }


   stddev_loop_j: for (j = 0; j < m; j++)
    {
      stddev[j] = 0.0;
      stddev_loop_i: for (i = 0; i < n; i++)
      {
#pragma HLS PIPELINE II=1
        stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
      }
      stddev[j] /= float_n;
      stddev[j] = sqrt(stddev[j]);


      stddev[j] = stddev[j] <= eps ? 1.0 : stddev[j];
    }


  norm_loop_i: for (i = 0; i < n; i++)
    norm_loop_j: for (j = 0; j < m; j++)
      {
#pragma HLS PIPELINE II=1
        data[i][j] -= mean[j];
        data[i][j] /= sqrt(float_n) * stddev[j];
      }


  corr_loop_i: for (i = 0; i < m-1; i++)
    {
      corr[i][i] = 1.0;
      corr_loop_j: for (j = i+1; j < m; j++)
        {
          double acc = 0.0;
          corr_loop_k: for (k = 0; k < n; k++)
          {
#pragma HLS PIPELINE II=1
            acc += (data[k][i] * data[k][j]);
          }
          corr[i][j] = acc;
          corr[j][i] = acc;
        }
    }
  corr[m-1][m-1] = 1.0;

}
}