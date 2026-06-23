#include "covariance.h"
#include <string.h>

#define TILE_N 256

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
  int t, ii;

  // Local mean buffer for reuse across phases
  double mean_l[M];
#pragma HLS ARRAY_PARTITION variable=mean_l cyclic factor=8 dim=1

  // Local row tile buffer: holds up to TILE_N rows of M columns
  double data_tile[TILE_N][M];

  // Accumulator for covariance (M x M); upper triangle accumulated incrementally
  double cov_acc[M][M];

  // ---- Phase 1: compute mean (tiled over rows) ----
  // Initialize mean accumulators
  double mean_acc[M];
  MEAN_INIT:
  for (j = 0; j < m; j++)
    {
#pragma HLS PIPELINE II=1
      mean_acc[j] = 0.0;
    }

  MEAN_TILE:
  for (t = 0; t < n; t += TILE_N)
    {
      int tn = (t + TILE_N <= n) ? TILE_N : (n - t);

      // --- load phase: bring a tile of rows into local buffer ---
      LOAD_MEAN:
      for (ii = 0; ii < tn; ii++)
        {
          memcpy(data_tile[ii], &data[t + ii][0], m * sizeof(double));
        }

      // --- compute phase: accumulate column sums from local tile ---
      MEAN_J:
      for (j = 0; j < m; j++)
        {
          double acc = mean_acc[j];
        MEAN_I:
          for (ii = 0; ii < tn; ii++)
            {
#pragma HLS PIPELINE II=1
              acc += data_tile[ii][j];
            }
          mean_acc[j] = acc;
        }
    }

  // Finalize mean
  MEAN_FIN:
  for (j = 0; j < m; j++)
    {
#pragma HLS PIPELINE II=1
      mean_l[j] = mean_acc[j] / float_n;
      mean[j] = mean_l[j];
    }

  // ---- Phase 2 + 3 combined: center data tile and accumulate covariance ----
  // Initialize covariance accumulators
  COV_INIT_I:
  for (i = 0; i < m; i++)
    {
    COV_INIT_J:
      for (j = 0; j < m; j++)
        {
#pragma HLS PIPELINE II=1
          cov_acc[i][j] = 0.0;
        }
    }

  COV_TILE:
  for (t = 0; t < n; t += TILE_N)
    {
      int tn = (t + TILE_N <= n) ? TILE_N : (n - t);

      // --- load phase: bring tile of rows into local buffer ---
      LOAD_COV:
      for (ii = 0; ii < tn; ii++)
        {
          memcpy(data_tile[ii], &data[t + ii][0], m * sizeof(double));
        }

      // --- center phase: subtract mean within local tile ---
      CENTER_I:
      for (ii = 0; ii < tn; ii++)
        {
        CENTER_J:
          for (j = 0; j < m; j++)
            {
#pragma HLS PIPELINE II=1
              data_tile[ii][j] -= mean_l[j];
            }
        }

      // --- compute phase: accumulate covariance products from local tile ---
      COV_I:
      for (i = 0; i < m; i++)
        {
        COV_J:
          for (j = i; j < m; j++)
            {
              double acc = cov_acc[i][j];
            COV_K:
              for (k = 0; k < tn; k++)
                {
#pragma HLS PIPELINE II=1
                  acc += data_tile[k][i] * data_tile[k][j];
                }
              cov_acc[i][j] = acc;
            }
        }
    }

  // ---- Store phase: finalize and write covariance to global memory ----
  STORE_I:
  for (i = 0; i < m; i++)
    {
    STORE_J:
      for (j = i; j < m; j++)
        {
#pragma HLS PIPELINE II=1
          double val = cov_acc[i][j] / (float_n - 1.0);
          cov[i][j] = val;
          cov[j][i] = val;
        }
    }

}
}