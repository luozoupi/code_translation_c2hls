#include "correlation.h"
#include <cstring>

// ---- Wide bus definitions (mc.h / ap_int substitute) ----
#define LARGE_BUS 512
// number of doubles packed in one wide bus word (512/64 = 8)
#define DOUBLES_PER_BUS (LARGE_BUS / 64)

// 512-bit wide bus word represented as a POD struct of 8 doubles
typedef struct {
  double v[DOUBLES_PER_BUS];
} MARS_WIDE_BUS_TYPE;

// ---- Wide-bus burst read of `num` doubles starting at element offset `base` ----
static void memcpy_wide_bus_read_double(double *local,
                                        MARS_WIDE_BUS_TYPE *bus,
                                        long base,
                                        int num)
{
  int words = (num + DOUBLES_PER_BUS - 1) / DOUBLES_PER_BUS;
  long word_base = base / DOUBLES_PER_BUS;
  long elem = 0;
  for (int w = 0; w < words; w++) {
#pragma HLS PIPELINE II=1
    MARS_WIDE_BUS_TYPE val = bus[word_base + w];
    for (int d = 0; d < DOUBLES_PER_BUS; d++) {
#pragma HLS UNROLL
      if (elem < num) {
        local[elem] = val.v[d];
      }
      elem++;
    }
  }
}

// ---- Wide-bus burst write of `num` doubles starting at element offset `base` ----
static void memcpy_wide_bus_write_double(MARS_WIDE_BUS_TYPE *bus,
                                         double *local,
                                         long base,
                                         int num)
{
  int words = (num + DOUBLES_PER_BUS - 1) / DOUBLES_PER_BUS;
  long word_base = base / DOUBLES_PER_BUS;
  long elem = 0;
  for (int w = 0; w < words; w++) {
#pragma HLS PIPELINE II=1
    MARS_WIDE_BUS_TYPE val;
    for (int d = 0; d < DOUBLES_PER_BUS; d++) {
#pragma HLS UNROLL
      val.v[d] = (elem < num) ? local[elem] : 0.0;
      elem++;
    }
    bus[word_base + w] = val;
  }
}

// Tile size: number of rows per tile
#define TILE 10

// ---- LOAD helper: stage one tile of rows into selected buffer ----
static void load_tile(MARS_WIDE_BUS_TYPE *data,
                      double buf1[TILE][M],
                      double buf2[TILE][M],
                      int tile_base,
                      int rows,
                      bool flag)
{
  for (int ii = 0; ii < TILE; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
    if (ii >= rows) break;
    double tmp[M];
#pragma HLS ARRAY_PARTITION variable=tmp cyclic factor=8 dim=1
    // burst read one row (M doubles) from global memory
    memcpy_wide_bus_read_double(tmp, data, (long)(tile_base + ii) * M, M);
    for (int j = 0; j < M; j++) {
#pragma HLS LOOP_TRIPCOUNT min=M max=M
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
      double v = tmp[j];
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

// ---- Wide-bus implementation core ----
static void kernel_correlation_wide(
			double float_n,
			MARS_WIDE_BUS_TYPE *data,
			MARS_WIDE_BUS_TYPE *corr,
			MARS_WIDE_BUS_TYPE *mean,
			MARS_WIDE_BUS_TYPE *stddev)
{
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
  // store mean and stddev via wide-bus burst writes
  memcpy_wide_bus_write_double(mean, mean_local, 0, m);
  memcpy_wide_bus_write_double(stddev, stddev_local, 0, m);

  // store normalized data, row by row using wide-bus bursts
  store_data:
  for (i = 0; i < n; i++) {
#pragma HLS LOOP_TRIPCOUNT min=N max=N
    double tmp[M];
#pragma HLS ARRAY_PARTITION variable=tmp cyclic factor=8 dim=1
    for (j = 0; j < m; j++) {
#pragma HLS LOOP_TRIPCOUNT min=M max=M
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
      tmp[j] = data_local[i][j];
    }
    memcpy_wide_bus_write_double(data, tmp, (long)i * M, m);
  }

  // store corr matrix, row by row using wide-bus bursts
  store_corr:
  for (i = 0; i < m; i++) {
#pragma HLS LOOP_TRIPCOUNT min=M max=M
    double tmp[M];
#pragma HLS ARRAY_PARTITION variable=tmp cyclic factor=8 dim=1
    for (j = 0; j < m; j++) {
#pragma HLS LOOP_TRIPCOUNT min=M max=M
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
      tmp[j] = corr_local[i][j];
    }
    memcpy_wide_bus_write_double(corr, tmp, (long)i * M, m);
  }
}

// ---- Top-level wrapper: matches header-declared signature ----
extern "C" {
void kernel_correlation( 
			double float_n,
			double data[ N + 0][M + 0],
			double corr[ M + 0][M + 0],
			double mean[ M + 0],
			double stddev[ M + 0])
{
#pragma HLS INTERFACE m_axi port=data   offset=slave bundle=gmem0 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=corr   offset=slave bundle=gmem1 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=mean   offset=slave bundle=gmem2 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=stddev offset=slave bundle=gmem3 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE s_axilite port=float_n bundle=control
#pragma HLS INTERFACE s_axilite port=data    bundle=control
#pragma HLS INTERFACE s_axilite port=corr    bundle=control
#pragma HLS INTERFACE s_axilite port=mean    bundle=control
#pragma HLS INTERFACE s_axilite port=stddev  bundle=control
#pragma HLS INTERFACE s_axilite port=return  bundle=control

  // Reinterpret the global-memory pointers as wide-bus words for coalesced
  // burst transfers. Layout is identical (contiguous doubles).
  kernel_correlation_wide(
      float_n,
      reinterpret_cast<MARS_WIDE_BUS_TYPE *>(data),
      reinterpret_cast<MARS_WIDE_BUS_TYPE *>(corr),
      reinterpret_cast<MARS_WIDE_BUS_TYPE *>(mean),
      reinterpret_cast<MARS_WIDE_BUS_TYPE *>(stddev));
}
}