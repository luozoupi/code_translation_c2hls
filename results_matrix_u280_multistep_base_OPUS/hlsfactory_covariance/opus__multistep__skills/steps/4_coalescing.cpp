#include "covariance.h"
#include <string.h>

#define TILE_N 256
#define LARGE_BUS 512
#define DOUBLES_PER_BUS (LARGE_BUS / 64)  // 8 doubles per 512-bit word

// Wide-bus word: packs 8 doubles into one 512-bit memory transaction.
typedef struct {
  double v[DOUBLES_PER_BUS];
} mars_wide_bus_t;

// ---- Wide-bus helper functions ----
// Read num doubles starting at double-index `d_off` from wide-bus memory into buffer.
static void memcpy_wide_bus_read_double(double *buffer,
                                        mars_wide_bus_t *bus,
                                        long d_off,
                                        int num_doubles)
{
  long word_base = d_off / DOUBLES_PER_BUS;
  int e0 = (int)(d_off % DOUBLES_PER_BUS);

  int produced = 0;
  long w = word_base;
  int e = e0;
READ_LOOP:
  while (produced < num_doubles)
    {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
      mars_wide_bus_t word = bus[w];
    READ_INNER:
      for (; e < DOUBLES_PER_BUS && produced < num_doubles; e++)
        {
#pragma HLS PIPELINE II=1
          buffer[produced] = word.v[e];
          produced++;
        }
      e = 0;
      w++;
    }
}

// Write num doubles from buffer into wide-bus memory starting at double-index `d_off`.
static void memcpy_wide_bus_write_double(mars_wide_bus_t *bus,
                                         double *buffer,
                                         long d_off,
                                         int num_doubles)
{
  long word_base = d_off / DOUBLES_PER_BUS;
  int e0 = (int)(d_off % DOUBLES_PER_BUS);

  int consumed = 0;
  long w = word_base;
  int e = e0;
WRITE_LOOP:
  while (consumed < num_doubles)
    {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
      mars_wide_bus_t word = bus[w]; // read-modify-write to preserve partial words
    WRITE_INNER:
      for (; e < DOUBLES_PER_BUS && consumed < num_doubles; e++)
        {
#pragma HLS PIPELINE II=1
          word.v[e] = buffer[consumed];
          consumed++;
        }
      bus[w] = word;
      e = 0;
      w++;
    }
}

// ---- Load helper: brings a tile of rows into the selected local buffer ----
static void load_tile(mars_wide_bus_t *data,
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
          memcpy_wide_bus_read_double(tile0[ii], data,
                                      (long)(t + ii) * M, m);
        }
    }
  else
    {
    LOAD_T1:
      for (int ii = 0; ii < tn; ii++)
        {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_N
          memcpy_wide_bus_read_double(tile1[ii], data,
                                      (long)(t + ii) * M, m);
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
#pragma HLS INTERFACE m_axi port=data offset=slave bundle=gmem0 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=cov  offset=slave bundle=gmem1 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=mean offset=slave bundle=gmem2 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE s_axilite port=float_n bundle=control
#pragma HLS INTERFACE s_axilite port=data    bundle=control
#pragma HLS INTERFACE s_axilite port=cov     bundle=control
#pragma HLS INTERFACE s_axilite port=mean    bundle=control
#pragma HLS INTERFACE s_axilite port=return  bundle=control

    const int n = N;
    const int m = M;

  // Reinterpret global-memory pointers as wide-bus words for coalesced access.
  mars_wide_bus_t *data_w = (mars_wide_bus_t *)data;
  mars_wide_bus_t *cov_w  = (mars_wide_bus_t *)cov;
  mars_wide_bus_t *mean_w = (mars_wide_bus_t *)mean;

  int i, j;

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
    load_tile(data_w, data_tile_0, data_tile_1, 0, tn0, m, false);
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
          load_tile(data_w, data_tile_0, data_tile_1, t_next, tn_next, m, !flag);
        }

      // Compute current tile from its buffer
      compute_mean(data_tile_0, data_tile_1, mean_acc, tn_cur, m, flag);
    }

  // Finalize mean
  double mean_out[M];
#pragma HLS ARRAY_PARTITION variable=mean_out cyclic factor=8 dim=1
  MEAN_FIN:
  for (j = 0; j < m; j++)
    {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
      mean_l[j] = mean_acc[j] / float_n;
      mean_out[j] = mean_l[j];
    }
  // Write mean to global memory via wide bus
  memcpy_wide_bus_write_double(mean_w, mean_out, 0, m);

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
    load_tile(data_w, data_tile_0, data_tile_1, 0, tn0, m, false);
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
          load_tile(data_w, data_tile_0, data_tile_1, t_next, tn_next, m, !flag);
        }

      // Compute current tile from its buffer (center + accumulate)
      compute_cov(data_tile_0, data_tile_1, cov_acc, mean_l, tn_cur, m, flag);
    }

  // ---- Store phase: finalize and write covariance to global memory ----
  // Use a local row buffer to enable wide-bus burst writes per row
  double cov_row[M];
#pragma HLS ARRAY_PARTITION variable=cov_row cyclic factor=8 dim=1

  STORE_I:
  for (i = 0; i < m; i++)
    {
      // Build full row i (both triangles) from cov_acc
    STORE_BUILD:
      for (j = 0; j < m; j++)
        {
#pragma HLS PIPELINE II=1
          double val;
          if (j >= i)
            val = cov_acc[i][j] / (float_n - 1.0);
          else
            val = cov_acc[j][i] / (float_n - 1.0);
          cov_row[j] = val;
        }
      // Burst write the full row via wide bus
      memcpy_wide_bus_write_double(cov_w, cov_row, (long)i * M, m);
    }

}
}