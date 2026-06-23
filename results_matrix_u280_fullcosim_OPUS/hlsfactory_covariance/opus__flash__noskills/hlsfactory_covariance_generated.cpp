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
#pragma HLS INTERFACE s_axilite port=data bundle=control
#pragma HLS INTERFACE s_axilite port=cov bundle=control
#pragma HLS INTERFACE s_axilite port=mean bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;
    const int m = M;

  int i, j, k;

  MEAN_J:for (j = 0; j < m; j++)
    {
      mean[j] = 0.0;
      MEAN_I:for (i = 0; i < n; i++)
      {
#pragma HLS PIPELINE II=1
        mean[j] += data[i][j];
      }
      mean[j] /= float_n;
    }

  CENTER_I:for (i = 0; i < n; i++)
    CENTER_J:for (j = 0; j < m; j++)
    {
#pragma HLS PIPELINE II=1
      data[i][j] -= mean[j];
    }

  COV_I:for (i = 0; i < m; i++)
    COV_J:for (j = i; j < m; j++)
      {
        cov[i][j] = 0.0;
        COV_K:for (k = 0; k < n; k++)
        {
#pragma HLS PIPELINE II=1
	  cov[i][j] += data[k][i] * data[k][j];
        }
        cov[i][j] /= (float_n - 1.0);
        cov[j][i] = cov[i][j];
      }

}
}