#include "correlation.h"
#include <cstring>

// Tile size: number of rows per tile
#define TILE 10

// ---- LOAD helper: stage one tile of rows into selected buffer ----
static void load_tile(double data[N][M],
                      double buf1[TILE][M],
                      double buf2[TILE][M],
                      int tile_base,
                      int rows,
                      bool flag)
{
  for (int ii = 0; ii < TILE; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
    if (ii >= rows) break;
    for (int j = 0; j < M; j++) {
#pragma HLS LOOP_TRIPCOUNT min=M max=M
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
      double v = data[tile_base + ii][j];
      if (flag) buf1[ii][j] = v;
      else      buf2[ii][j] = v;
    }
  }
}

// ---- COMPUTE helper: copy selected tile into full data_local ----
static void compute_tile(double data_local[N][M],
                         double buf1[TILE][M],
                         double buf2[TILE][M],
                         int tile_base,
                         int rows,
                         bool flag)
{
  for (int ii = 0; ii < TILE; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
    if (ii >= rows) break;
    for (int j = 0; j < M; j++) {
#pragma HLS LOOP_TRIPCOUNT min=M max=M
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
      double v = flag ? buf1[ii][j] : buf2[ii][j];
      data_local[tile_base + ii][j] = v;
    }
  }
}

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
#pragma HLS ARRAY_PARTITION variable=data_local cyclic factor=8 dim=2
  double mean_local[M];
#pragma HLS ARRAY_PARTITION variable=mean_local cyclic factor=8 dim=1
  double stddev_local[M];
#pragma HLS ARRAY_PARTITION variable=stddev_local cyclic factor=8 dim=1
  double corr_local[M][M];
#pragma HLS ARRAY_PARTITION variable=corr_local cyclic factor=8 dim=2

  // ---- Double buffers (ping-pong) for the LOAD phase ----
  double data_buf1[TILE][M];
#pragma HLS ARRAY_PARTITION variable=data_buf1 cyclic factor=8 dim=2
  double data_buf2[TILE][M];
#pragma HLS ARRAY_PARTITION variable=data_buf2 cyclic factor=8 dim=2

  // ---- LOAD phase with double buffering: overlap load(k+1) and compute(k) ----
  const int num_tiles = (n + TILE - 1) / TILE;

  load_double_buffer:
  for (int t = 0; t < num_tiles + 1; t++) {
#pragma HLS LOOP_TRIPCOUNT min=11 max=11
    bool load_flag = (t % 2 == 0);            // buffer to load INTO this iteration
    bool comp_flag = ((t - 1) % 2 == 0);      // buffer to consume from previous load

    int load_base = t * TILE;
    int load_rows = (load_base + TILE <= n) ? TILE : (n - load_base);
    if (load_base >= n) load_rows = 0;

    int comp_base = (t - 1) * TILE;
    int comp_rows = (comp_base + TILE <= n) ? TILE : (n - comp_base);
    if (t == 0 || comp_base >= n) comp_rows = 0;

    // Load tile t (producer) and compute/copy tile t-1 (consumer) overlap
    if (load_rows > 0)
      load_tile(data, data_buf1, data_buf2, load_base, load_rows, load_flag);

    if (comp_rows > 0)
      compute_tile(data_local, data_buf1, data_buf2, comp_base, comp_rows, comp_flag);
  }

  // ---- COMPUTE phase: mean ----
  compute_mean:
  for (j = 0; j < m; j++)
    {
#pragma HLS LOOP_TRIPCOUNT min=M max=M
      mean_local[j] = 0.0;
      for (i = 0; i < n; i++) {
#pragma HLS LOOP_TRIPCOUNT min=N max=N
#pragma HLS PIPELINE II=1
	mean_local[j] += data_local[i][j];
      }
      mean_local[j] /= float_n;
    }

  // ---- COMPUTE phase: stddev ----
  compute_stddev:
   for (j = 0; j < m; j++)
    {
#pragma HLS LOOP_TRIPCOUNT min=M max=M
      stddev_local[j] = 0.0;
      for (i = 0; i < n; i++) {
#pragma HLS LOOP_TRIPCOUNT min=N max=N
#pragma HLS PIPELINE II=1
        stddev_local[j] += (data_local[i][j] - mean_local[j]) * (data_local[i][j] - mean_local[j]);
      }
      stddev_local[j] /= float_n;
      stddev_local[j] = sqrt(stddev_local[j]);

      stddev_local[j] = stddev_local[j] <= eps ? 1.0 : stddev_local[j];
    }

  // ---- COMPUTE phase: center and normalize data ----
  compute_normalize:
  for (i = 0; i < n; i++) {
#pragma HLS LOOP_TRIPCOUNT min=N max=N
    for (j = 0; j < m; j++)
      {
#pragma HLS LOOP_TRIPCOUNT min=M max=M
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
        data_local[i][j] -= mean_local[j];
        data_local[i][j] /= sqrt(float_n) * stddev_local[j];
      }
  }

  // ---- COMPUTE phase: correlation ----
  compute_corr:
  for (i = 0; i < m-1; i++)
    {
#pragma HLS LOOP_TRIPCOUNT min=M max=M
      corr_local[i][i] = 1.0;
      for (j = i+1; j < m; j++)
        {
#pragma HLS LOOP_TRIPCOUNT min=M max=M
          double acc = 0.0;
          for (k = 0; k < n; k++) {
#pragma HLS LOOP_TRIPCOUNT min=N max=N
#pragma HLS PIPELINE II=1
            acc += (data_local[k][i] * data_local[k][j]);
          }
          corr_local[i][j] = acc;
          corr_local[j][i] = acc;
        }
    }
  corr_local[m-1][m-1] = 1.0;

  // ---- STORE phase: write results back to global memory ----
  store_mean:
  for (j = 0; j < m; j++) {
#pragma HLS LOOP_TRIPCOUNT min=M max=M
#pragma HLS PIPELINE II=1
    mean[j] = mean_local[j];
    stddev[j] = stddev_local[j];
  }

  store_data:
  for (i = 0; i < n; i++) {
#pragma HLS LOOP_TRIPCOUNT min=N max=N
    for (j = 0; j < m; j++) {
#pragma HLS LOOP_TRIPCOUNT min=M max=M
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
      data[i][j] = data_local[i][j];
    }
  }

  store_corr:
  for (i = 0; i < m; i++) {
#pragma HLS LOOP_TRIPCOUNT min=M max=M
    for (j = 0; j < m; j++) {
#pragma HLS LOOP_TRIPCOUNT min=M max=M
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
      corr[i][j] = corr_local[i][j];
    }
  }

}