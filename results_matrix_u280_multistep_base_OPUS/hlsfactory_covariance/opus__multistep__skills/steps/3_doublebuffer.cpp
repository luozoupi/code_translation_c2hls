#include "covariance.h"
#include <string.h>

#define TILE_N 256

// ---- Load helper: brings a tile of rows into the selected local buffer ----
static void load_tile(double data[N + 0][M + 0],
                      double tile0[TILE_N][M],
                      double tile1[TILE_N][M],
                      int t, int tn, int m, bool flag)
{
  if (!flag)
    {
    LOAD_T0:
      for (int ii = 0; ii < tn; ii++)
        {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_N
          memcpy(tile0[ii], &data[t + ii][0], m * sizeof(double));
        }
    }
  else
    {
    LOAD_T1:
      for (int ii = 0; ii < tn; ii++)
        {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_N
          memcpy(tile1[ii], &data[t + ii][0], m * sizeof(double));
        }
    }
}

// ---- Mean compute helper: accumulate column sums from selected buffer ----
static void compute_mean(double tile0[TILE_N][M],
                         double tile1[TILE_N][M],
                         double mean_acc[M],
                         int tn, int m, bool flag)
{
  if (!flag)
    {
    MEAN_I0:
      for (int ii = 0; ii < tn; ii++)
        {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_N
        MEAN_J0:
          for (int j = 0; j < m; j++)
            {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
              mean_acc[j] += tile0[ii][j];
            }
        }
    }
  else
    {
    MEAN_I1:
      for (int ii = 0; ii < tn; ii++)
        {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_N
        MEAN_J1:
          for (int j = 0; j < m; j++)
            {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
              mean_acc[j] += tile1[ii][j];
            }
        }
    }
}

// ---- Covariance compute helper: center + accumulate products ----
static void compute_cov(double tile0[TILE_N][M],
                        double tile1[TILE_N][M],
                        double cov_acc[M][M],
                        double mean_l[M],
                        int tn, int m, bool flag)
{
  if (!flag)
    {
      // center
    CENTER_I0:
      for (int ii = 0; ii < tn; ii++)
        {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_N
        CENTER_J0:
          for (int j = 0; j < m; j++)
            {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
              tile0[ii][j] -= mean_l[j];
            }
        }
      // covariance products
    COV_I0:
      for (int i = 0; i < m; i++)
        {
        COV_J0:
          for (int j = i; j < m; j++)
            {
              double acc = cov_acc[i][j];
            COV_K0:
              for (int k = 0; k < tn; k++)
                {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_N
#pragma HLS DEPENDENCE variable=tile0 inter false
                  acc += tile0[k][i] * tile0[k][j];
                }
              cov_acc[i][j] = acc;
            }
        }
    }
  else
    {
      // center
    CENTER_I1:
      for (int ii = 0; ii < tn; ii++)
        {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_N
        CENTER_J1:
          for (int j = 0; j < m; j++)
            {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
              tile1[ii][j] -= mean_l[j];
            }
        }
      // covariance products
    COV_I1:
      for (int i = 0; i < m; i++)
        {
        COV_J1:
          for (int j = i; j < m; j++)
            {
              double acc = cov_acc[i][j];
            COV_K1:
              for (int k = 0; k < tn; k++)
                {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_N
#pragma HLS DEPENDENCE variable=tile1 inter false
                  acc += tile1[k][i] * tile1[k][j];
                }
              cov_acc[i][j] = acc;
            }
        }
    }
}

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

  int i, j;
  int t;

  // Local mean buffer for reuse across phases
  double mean_l[M];
#pragma HLS ARRAY_PARTITION variable=mean_l cyclic factor=8 dim=1

  // Double-buffered row tiles: two copies for ping-pong load/compute overlap
  double data_tile_0[TILE_N][M];
  double data_tile_1[TILE_N][M];
#pragma HLS ARRAY_PARTITION variable=data_tile_0 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=data_tile_1 cyclic factor=8 dim=2

  // Accumulator for covariance (M x M); upper triangle accumulated incrementally
  double cov_acc[M][M];
#pragma HLS ARRAY_PARTITION variable=cov_acc cyclic factor=8 dim=2

  // ---- Phase 1: compute mean (tiled over rows, double buffered) ----
  double mean_acc[M];
#pragma HLS ARRAY_PARTITION variable=mean_acc cyclic factor=8 dim=1
  MEAN_INIT:
  for (j = 0; j < m; j++)
    {
#pragma HLS PIPELINE II=1
      mean_acc[j] = 0.0;
    }

  // Number of tiles
  int num_tiles = (n + TILE_N - 1) / TILE_N;

  // Prologue: load first tile into buffer 0
  {
    int tn0 = (TILE_N <= n) ? TILE_N : n;
    load_tile(data, data_tile_0, data_tile_1, 0, tn0, m, false);
  }

  // Software-pipelined mean loop: compute tile k while loading tile k+1
  MEAN_TILE:
  for (int tile = 0; tile < num_tiles; tile++)
    {
#pragma HLS LOOP_TRIPCOUNT min=1 max=1
      int t_cur = tile * TILE_N;
      int tn_cur = (t_cur + TILE_N <= n) ? TILE_N : (n - t_cur);
      bool flag = (tile % 2) != 0; // false -> buffer0, true -> buffer1

      // Load next tile into the OTHER buffer
      if (tile + 1 < num_tiles)
        {
          int t_next = (tile + 1) * TILE_N;
          int tn_next = (t_next + TILE_N <= n) ? TILE_N : (n - t_next);
          load_tile(data, data_tile_0, data_tile_1, t_next, tn_next, m, !flag);
        }

      // Compute current tile from its buffer
      compute_mean(data_tile_0, data_tile_1, mean_acc, tn_cur, m, flag);
    }

  // Finalize mean
  MEAN_FIN:
  for (j = 0; j < m; j++)
    {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
      mean_l[j] = mean_acc[j] / float_n;
      mean[j] = mean_l[j];
    }

  // ---- Phase 2 + 3: center data and accumulate covariance (double buffered) ----
  COV_INIT_I:
  for (i = 0; i < m; i++)
    {
    COV_INIT_J:
      for (j = 0; j < m; j++)
        {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
          cov_acc[i][j] = 0.0;
        }
    }

  // Prologue: load first tile into buffer 0
  {
    int tn0 = (TILE_N <= n) ? TILE_N : n;
    load_tile(data, data_tile_0, data_tile_1, 0, tn0, m, false);
  }

  // Software-pipelined covariance loop
  COV_TILE:
  for (int tile = 0; tile < num_tiles; tile++)
    {
#pragma HLS LOOP_TRIPCOUNT min=1 max=1
      int t_cur = tile * TILE_N;
      int tn_cur = (t_cur + TILE_N <= n) ? TILE_N : (n - t_cur);
      bool flag = (tile % 2) != 0;

      // Load next tile into the OTHER buffer
      if (tile + 1 < num_tiles)
        {
          int t_next = (tile + 1) * TILE_N;
          int tn_next = (t_next + TILE_N <= n) ? TILE_N : (n - t_next);
          load_tile(data, data_tile_0, data_tile_1, t_next, tn_next, m, !flag);
        }

      // Compute current tile from its buffer (center + accumulate)
      compute_cov(data_tile_0, data_tile_1, cov_acc, mean_l, tn_cur, m, flag);
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