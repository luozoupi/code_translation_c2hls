#include "gemver.h"
#include <string.h>

#define TILE 8
#define LARGE_BUS 512

// Number of doubles per wide-bus word (512 bits / 64 bits).
#define WIDE_FACTOR (LARGE_BUS / 64)

// Wide-bus word: a packed group of WIDE_FACTOR doubles (512 bits total).
typedef struct {
  double data[WIDE_FACTOR];
} MARS_WIDE_BUS_TYPE;

// ---- Wide-bus helper functions (self-contained) ----
// Read `num` doubles starting at byte offset `offset_bytes` from wide bus `bus` into `local`.
static void memcpy_wide_bus_read_double(double *local, MARS_WIDE_BUS_TYPE *bus,
                                        long offset_bytes, int num) {
  long base_word = offset_bytes / (LARGE_BUS / 8);
  int  base_elem = (int)((offset_bytes % (LARGE_BUS / 8)) / 8);

  int produced = 0;
  long w = base_word;
  int elem = base_elem;
  rd_loop: while (produced < num) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=2048
    MARS_WIDE_BUS_TYPE word = bus[w];
    rd_inner: for (int k = elem; k < WIDE_FACTOR && produced < num; k++) {
#pragma HLS PIPELINE II=1
      local[produced] = word.data[k];
      produced++;
    }
    elem = 0;
    w++;
  }
}

// Write `num` doubles from `local` into wide bus `bus` at byte offset `offset_bytes`.
static void memcpy_wide_bus_write_double(MARS_WIDE_BUS_TYPE *bus, double *local,
                                         long offset_bytes, int num) {
  long base_word = offset_bytes / (LARGE_BUS / 8);
  int  base_elem = (int)((offset_bytes % (LARGE_BUS / 8)) / 8);

  int consumed = 0;
  long w = base_word;
  int elem = base_elem;
  wr_loop: while (consumed < num) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=2048
    MARS_WIDE_BUS_TYPE word = bus[w];
    wr_inner: for (int k = elem; k < WIDE_FACTOR && consumed < num; k++) {
#pragma HLS PIPELINE II=1
      word.data[k] = local[consumed];
      consumed++;
    }
    bus[w] = word;
    elem = 0;
    w++;
  }
}

// Load a tile of A rows from global memory (wide bus) into one of the ping-pong buffers.
static void load_tile(MARS_WIDE_BUS_TYPE *A,
                      double buf0[TILE][N], double buf1[TILE][N],
                      int ti, int rows, int flag) {
  if (flag == 0) {
    load0: for (int r = 0; r < rows; r++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
      memcpy_wide_bus_read_double(&buf0[r][0], A, (long)(ti + r) * N * sizeof(double), N);
    }
  } else {
    load1: for (int r = 0; r < rows; r++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
      memcpy_wide_bus_read_double(&buf1[r][0], A, (long)(ti + r) * N * sizeof(double), N);
    }
  }
}

// Compute: copy staged tile rows into the full local matrix l_A.
static void compute_tile(double l_A[N][N],
                         double buf0[TILE][N], double buf1[TILE][N],
                         int ti, int rows, int flag) {
  if (flag == 0) {
    comp0: for (int r = 0; r < rows; r++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
      copy0: for (int c = 0; c < N; c++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
        l_A[ti + r][c] = buf0[r][c];
      }
    }
  } else {
    comp1: for (int r = 0; r < rows; r++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
      copy1: for (int c = 0; c < N; c++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
        l_A[ti + r][c] = buf1[r][c];
      }
    }
  }
}

// Wide-bus implementation (separate name to avoid conflict with header declaration).
static void kernel_gemver_wide(
		   double alpha,
		   double beta,
		   MARS_WIDE_BUS_TYPE *A,
		   MARS_WIDE_BUS_TYPE *u1,
		   MARS_WIDE_BUS_TYPE *v1,
		   MARS_WIDE_BUS_TYPE *u2,
		   MARS_WIDE_BUS_TYPE *v2,
		   MARS_WIDE_BUS_TYPE *w,
		   MARS_WIDE_BUS_TYPE *x,
		   MARS_WIDE_BUS_TYPE *y,
		   MARS_WIDE_BUS_TYPE *z)
{
    const int n = N;

  int i, j, ti;

  // Stage vectors into local buffers for fast reuse across the loop nests.
  double l_u1[N], l_v1[N], l_u2[N], l_v2[N];
  double l_x[N], l_y[N], l_z[N], l_w[N];
#pragma HLS ARRAY_PARTITION variable=l_v1 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_v2 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_x  cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_y  cyclic factor=8 dim=1

  // Local tile/buffer for the full matrix A (staged once, reused across phases).
  static double l_A[N][N];
#pragma HLS ARRAY_PARTITION variable=l_A cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_A cyclic factor=8 dim=1

  // ---- Ping-pong tile buffers for double-buffered staging of A ----
  static double tileA_0[TILE][N];
  static double tileA_1[TILE][N];
#pragma HLS ARRAY_PARTITION variable=tileA_0 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=tileA_1 cyclic factor=8 dim=2

  // Load all vectors via wide bus burst into local buffers.
  memcpy_wide_bus_read_double(&l_u1[0], u1, 0, N);
  memcpy_wide_bus_read_double(&l_v1[0], v1, 0, N);
  memcpy_wide_bus_read_double(&l_u2[0], u2, 0, N);
  memcpy_wide_bus_read_double(&l_v2[0], v2, 0, N);
  memcpy_wide_bus_read_double(&l_x[0],  x,  0, N);
  memcpy_wide_bus_read_double(&l_y[0],  y,  0, N);
  memcpy_wide_bus_read_double(&l_z[0],  z,  0, N);
  memcpy_wide_bus_read_double(&l_w[0],  w,  0, N);

  // ---- DOUBLE-BUFFERED LOAD PHASE: stage matrix A tile by tile ----
  // Load of tile k+1 overlaps with the staging (compute) of tile k.
  int num_tiles = (n + TILE - 1) / TILE;

  // Prologue: load first tile.
  {
    int rows0 = (TILE <= n) ? TILE : n;
    load_tile(A, tileA_0, tileA_1, 0, rows0, 0);
  }

  stage_A: for (int t = 0; t < num_tiles; t++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=15
    int ti_cur  = t * TILE;
    int rows_cur = (ti_cur + TILE <= n) ? TILE : (n - ti_cur);
    int flag_cur = t & 1;

    // Load next tile into the OTHER buffer (overlaps with compute of current).
    if (t + 1 < num_tiles) {
      int ti_next  = (t + 1) * TILE;
      int rows_next = (ti_next + TILE <= n) ? TILE : (n - ti_next);
      load_tile(A, tileA_0, tileA_1, ti_next, rows_next, (t + 1) & 1);
    }

    // Compute (stage current tile into full l_A) from current buffer.
    compute_tile(l_A, tileA_0, tileA_1, ti_cur, rows_cur, flag_cur);
  }

  // ---- COMPUTE PHASE (operates entirely on local buffers) ----

  // Phase 1: A = A + u1*v1^T + u2*v2^T
  phase1_i: for (i = 0; i < n; i++) {
    double u1i = l_u1[i];
    double u2i = l_u2[i];
    phase1_j: for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS DEPENDENCE variable=l_A inter false
      l_A[i][j] = l_A[i][j] + u1i * l_v1[j] + u2i * l_v2[j];
    }
  }

  // Phase 2: x = x + beta * A^T * y
  phase2_i: for (i = 0; i < n; i++) {
    double acc = l_x[i];
    phase2_j: for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS DEPENDENCE variable=l_A inter false
      acc += beta * l_A[j][i] * l_y[j];
    }
    l_x[i] = acc;
  }

  // Phase 3: x = x + z
  phase3: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
    l_x[i] = l_x[i] + l_z[i];
  }

  // Phase 4: w = w + alpha * A * x
  phase4_i: for (i = 0; i < n; i++) {
    double acc = l_w[i];
    phase4_j: for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS DEPENDENCE variable=l_A inter false
      acc += alpha * l_A[i][j] * l_x[j];
    }
    l_w[i] = acc;
  }

  // ---- STORE PHASE: write back modified matrix A (tile by tile) and result vectors ----
  store_A_rows: for (int r = 0; r < n; r++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=120
    memcpy_wide_bus_write_double(A, &l_A[r][0], (long)r * N * sizeof(double), N);
  }

  // Write back result vectors via wide bus burst.
  memcpy_wide_bus_write_double(x, &l_x[0], 0, N);
  memcpy_wide_bus_write_double(w, &l_w[0], 0, N);
}

// Top-level wrapper matching the header-declared signature.
// Pointers are reinterpreted as wide-bus words for coalesced burst access.
extern "C" {
void kernel_gemver(
		   double alpha,
		   double beta,
		   double A[ N + 0][N + 0],
		   double u1[ N + 0],
		   double v1[ N + 0],
		   double u2[ N + 0],
		   double v2[ N + 0],
		   double w[ N + 0],
		   double x[ N + 0],
		   double y[ N + 0],
		   double z[ N + 0])
{
#pragma HLS INTERFACE m_axi port=A   offset=slave bundle=gmem0 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=u1  offset=slave bundle=gmem1 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=v1  offset=slave bundle=gmem2 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=u2  offset=slave bundle=gmem3 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=v2  offset=slave bundle=gmem4 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=w   offset=slave bundle=gmem5 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=x   offset=slave bundle=gmem6 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=y   offset=slave bundle=gmem7 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=z   offset=slave bundle=gmem8 max_read_burst_length=256 max_write_burst_length=256

#pragma HLS INTERFACE s_axilite port=alpha bundle=control
#pragma HLS INTERFACE s_axilite port=beta  bundle=control
#pragma HLS INTERFACE s_axilite port=A     bundle=control
#pragma HLS INTERFACE s_axilite port=u1    bundle=control
#pragma HLS INTERFACE s_axilite port=v1    bundle=control
#pragma HLS INTERFACE s_axilite port=u2    bundle=control
#pragma HLS INTERFACE s_axilite port=v2    bundle=control
#pragma HLS INTERFACE s_axilite port=w     bundle=control
#pragma HLS INTERFACE s_axilite port=x     bundle=control
#pragma HLS INTERFACE s_axilite port=y     bundle=control
#pragma HLS INTERFACE s_axilite port=z     bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  kernel_gemver_wide(alpha, beta,
                     (MARS_WIDE_BUS_TYPE *)A,
                     (MARS_WIDE_BUS_TYPE *)u1,
                     (MARS_WIDE_BUS_TYPE *)v1,
                     (MARS_WIDE_BUS_TYPE *)u2,
                     (MARS_WIDE_BUS_TYPE *)v2,
                     (MARS_WIDE_BUS_TYPE *)w,
                     (MARS_WIDE_BUS_TYPE *)x,
                     (MARS_WIDE_BUS_TYPE *)y,
                     (MARS_WIDE_BUS_TYPE *)z);
}
}