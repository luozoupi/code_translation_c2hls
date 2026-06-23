#include "bicg.h"
#include <cstring>

#define TILE_SIZE 256

// ---- Wide bus definitions (inlined; ap_int.h / mc.h not available) ----
#define LARGE_BUS 512
#define WIDE_DOUBLES (LARGE_BUS / 64)  // 8 doubles per 512-bit word

// 512-bit wide bus word modeled as a packed struct of 8 doubles.
typedef struct {
  double data[WIDE_DOUBLES];
} MARS_WIDE_BUS_TYPE;

// Read `bytes` bytes (a multiple of sizeof(double)) from wide-bus `src`
// starting at byte offset `offset_bytes` into local double buffer `dst`.
static void memcpy_wide_bus_read_float(double *dst,
                                       MARS_WIDE_BUS_TYPE *src,
                                       long offset_bytes,
                                       long bytes)
{
  long num_elems = bytes / sizeof(double);
  long base_elem = offset_bytes / sizeof(double);
  long widx = base_elem / WIDE_DOUBLES;
  long woff = base_elem % WIDE_DOUBLES;

  read_loop:
  for (long e = 0; e < num_elems; e++) {
#pragma HLS PIPELINE II=1
    dst[e] = src[widx].data[woff];
    woff++;
    if (woff == WIDE_DOUBLES) { woff = 0; widx++; }
  }
}

// Write `bytes` bytes (a multiple of sizeof(double)) from local double buffer
// `src` to wide-bus `dst` starting at byte offset `offset_bytes`.
static void memcpy_wide_bus_write_float(MARS_WIDE_BUS_TYPE *dst,
                                        double *src,
                                        long offset_bytes,
                                        long bytes)
{
  long num_elems = bytes / sizeof(double);
  long base_elem = offset_bytes / sizeof(double);
  long widx = base_elem / WIDE_DOUBLES;
  long woff = base_elem % WIDE_DOUBLES;

  write_loop:
  for (long e = 0; e < num_elems; e++) {
#pragma HLS PIPELINE II=1
    dst[widx].data[woff] = src[e];
    woff++;
    if (woff == WIDE_DOUBLES) { woff = 0; widx++; }
  }
}
// ---- end wide bus helpers ----

static void load_A_row(MARS_WIDE_BUS_TYPE *A,
                       double A_tile_1[TILE_SIZE],
                       double A_tile_2[TILE_SIZE],
                       int i, int m, int flag)
{
  // Read one row (m elements) of A starting at offset i*M
  if (flag == 0)
    memcpy_wide_bus_read_float(A_tile_1, A, (long)i * M * sizeof(double), (long)m * sizeof(double));
  else
    memcpy_wide_bus_read_float(A_tile_2, A, (long)i * M * sizeof(double), (long)m * sizeof(double));
}

static void compute_A_row(double A_tile_1[TILE_SIZE],
                          double A_tile_2[TILE_SIZE],
                          double s_local[M],
                          double p_local[M],
                          double q_local[N],
                          double r_i, int i, int m, int flag)
{
  double q_acc = 0.0;
  compute_tile:
  for (int j = 0; j < m; j++) {
#pragma HLS LOOP_TRIPCOUNT min=116 max=116
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS DEPENDENCE variable=s_local inter false
    double a = (flag == 0) ? A_tile_1[j] : A_tile_2[j];
    s_local[j] = s_local[j] + r_i * a;
    q_acc = q_acc + a * p_local[j];
  }
  q_local[i] = q_acc;
}

// Wide-bus implementation of the kernel.
static void kernel_bicg_wide(
		 MARS_WIDE_BUS_TYPE *A,
		 MARS_WIDE_BUS_TYPE *s,
		 MARS_WIDE_BUS_TYPE *q,
		 MARS_WIDE_BUS_TYPE *p,
		 MARS_WIDE_BUS_TYPE *r)
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=s offset=slave bundle=gmem1 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=q offset=slave bundle=gmem2 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=p offset=slave bundle=gmem3 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=r offset=slave bundle=gmem4 max_read_burst_length=256 max_write_burst_length=256

    const int n = N;
    const int m = M;

  int i, j;

  // Local buffers for the full M-dimension working set (s and p reused across all rows)
  double s_local[M];
#pragma HLS ARRAY_PARTITION variable=s_local cyclic factor=4 dim=1
  double p_local[M];
#pragma HLS ARRAY_PARTITION variable=p_local cyclic factor=4 dim=1

  // Double-buffered local buffers for one tile of an A row
  double A_tile_1[TILE_SIZE];
#pragma HLS ARRAY_PARTITION variable=A_tile_1 cyclic factor=4 dim=1
  double A_tile_2[TILE_SIZE];
#pragma HLS ARRAY_PARTITION variable=A_tile_2 cyclic factor=4 dim=1

  // Local buffers for q and r (load r, accumulate q)
  double q_local[N];
#pragma HLS ARRAY_PARTITION variable=q_local cyclic factor=4 dim=1
  double r_local[N];
#pragma HLS ARRAY_PARTITION variable=r_local cyclic factor=4 dim=1

  // ---------- LOAD phase: p and r into local buffers, init s_local ----------
  // Burst-read p into local buffer using wide bus
  memcpy_wide_bus_read_float(p_local, p, 0, (long)m * sizeof(double));

  init_s:
  for (j = 0; j < m; j++) {
#pragma HLS LOOP_TRIPCOUNT min=116 max=116
#pragma HLS PIPELINE II=1
    s_local[j] = 0.0;
  }

  // Burst-read r into local buffer using wide bus
  memcpy_wide_bus_read_float(r_local, r, 0, (long)n * sizeof(double));

  // ---------- COMPUTE phase: double-buffered over rows ----------
  // Prologue: load first row's tile into buffer 0
  load_A_row(A, A_tile_1, A_tile_2, 0, m, 0);

  outer_n:
  for (i = 0; i < n; i++)
    {
#pragma HLS LOOP_TRIPCOUNT min=124 max=124
      int flag = i % 2;  // buffer holding the row currently being computed

      // Load next row's tile into the OTHER buffer while we compute this one
      if (i + 1 < n) {
        load_A_row(A, A_tile_1, A_tile_2, i + 1, m, (i + 1) % 2);
      }

      // Compute on the current row's tile
      compute_A_row(A_tile_1, A_tile_2, s_local, p_local, q_local,
                    r_local[i], i, m, flag);
    }

  // ---------- STORE phase: write back q and s ----------
  // Burst-write q from local buffer using wide bus
  memcpy_wide_bus_write_float(q, q_local, 0, (long)n * sizeof(double));

  // Burst-write s from local buffer using wide bus
  memcpy_wide_bus_write_float(s, s_local, 0, (long)m * sizeof(double));
}

// Top function matching the header declaration exactly. Casts the
// double-typed AXI pointers to the wide-bus type and forwards to the
// coalesced implementation.
extern "C" {
void kernel_bicg(
		 double A[ N + 0][M + 0],
		 double s[ M + 0],
		 double q[ N + 0],
		 double p[ M + 0],
		 double r[ N + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=s offset=slave bundle=gmem1 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=q offset=slave bundle=gmem2 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=p offset=slave bundle=gmem3 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=r offset=slave bundle=gmem4 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=s bundle=control
#pragma HLS INTERFACE s_axilite port=q bundle=control
#pragma HLS INTERFACE s_axilite port=p bundle=control
#pragma HLS INTERFACE s_axilite port=r bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  kernel_bicg_wide(
      reinterpret_cast<MARS_WIDE_BUS_TYPE *>(A),
      reinterpret_cast<MARS_WIDE_BUS_TYPE *>(s),
      reinterpret_cast<MARS_WIDE_BUS_TYPE *>(q),
      reinterpret_cast<MARS_WIDE_BUS_TYPE *>(p),
      reinterpret_cast<MARS_WIDE_BUS_TYPE *>(r));
}
}