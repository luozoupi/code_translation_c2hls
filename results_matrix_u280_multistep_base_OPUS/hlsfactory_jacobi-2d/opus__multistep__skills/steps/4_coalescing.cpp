#include "jacobi-2d.h"
#include <cstring>

// ---- Wide-bus definitions (self-contained, no Xilinx headers required) ----
// 512-bit bus = 8 doubles per wide word.
#define LARGE_BUS 512
#define WIDE_BUS_DOUBLES (LARGE_BUS / 64)

typedef struct {
  double data[WIDE_BUS_DOUBLES];
} MARS_WIDE_BUS_TYPE;

// Read 'num_bytes' bytes from wide bus 'src' (starting at byte offset) into local double array
static void memcpy_wide_bus_read_float(double *local, MARS_WIDE_BUS_TYPE *src,
                                       long byte_offset, long num_bytes)
{
  long num_elem = num_bytes / (long)sizeof(double);
  long elem_offset = byte_offset / (long)sizeof(double);
  for (long e = 0; e < num_elem; e++) {
#pragma HLS PIPELINE II=1
    long global_idx = elem_offset + e;
    long word_idx = global_idx / WIDE_BUS_DOUBLES;
    int sub = (int)(global_idx % WIDE_BUS_DOUBLES);
    local[e] = src[word_idx].data[sub];
  }
}

// Write 'num_bytes' bytes from local double array into wide bus 'dst' (starting at byte offset)
static void memcpy_wide_bus_write_float(MARS_WIDE_BUS_TYPE *dst, double *local,
                                        long byte_offset, long num_bytes)
{
  long num_elem = num_bytes / (long)sizeof(double);
  long elem_offset = byte_offset / (long)sizeof(double);
  for (long e = 0; e < num_elem; e++) {
#pragma HLS PIPELINE II=1
    long global_idx = elem_offset + e;
    long word_idx = global_idx / WIDE_BUS_DOUBLES;
    int sub = (int)(global_idx % WIDE_BUS_DOUBLES);
    dst[word_idx].data[sub] = local[e];
  }
}

// Load phase: stage global data into the selected buffer set
static void load(MARS_WIDE_BUS_TYPE *A, MARS_WIDE_BUS_TYPE *B,
                 double A_local_1[N][N], double B_local_1[N][N],
                 double A_local_2[N][N], double B_local_2[N][N],
                 int flag)
{
  const int n = N;
  if (flag == 0) {
    for (int i = 0; i < n; i++) {
#pragma HLS LOOP_TRIPCOUNT min=N max=N
      memcpy_wide_bus_read_float(&A_local_1[i][0], A, (long)i * n * sizeof(double), n * sizeof(double));
      memcpy_wide_bus_read_float(&B_local_1[i][0], B, (long)i * n * sizeof(double), n * sizeof(double));
    }
  } else {
    for (int i = 0; i < n; i++) {
#pragma HLS LOOP_TRIPCOUNT min=N max=N
      memcpy_wide_bus_read_float(&A_local_2[i][0], A, (long)i * n * sizeof(double), n * sizeof(double));
      memcpy_wide_bus_read_float(&B_local_2[i][0], B, (long)i * n * sizeof(double), n * sizeof(double));
    }
  }
}

// Compute phase: run all timesteps on the selected buffer set
static void compute(double A_local_1[N][N], double B_local_1[N][N],
                    double A_local_2[N][N], double B_local_2[N][N],
                    int flag)
{
  const int n = N;
  const int tsteps = TSTEPS;
  int t, i, j;

  for (t = 0; t < tsteps; t++)
    {
#pragma HLS LOOP_TRIPCOUNT min=TSTEPS max=TSTEPS
      for (i = 1; i < n - 1; i++)
	for (j = 1; j < n - 1; j++)
	{
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS DEPENDENCE variable=B_local_1 inter false
#pragma HLS DEPENDENCE variable=B_local_2 inter false
	  if (flag == 0) {
	    B_local_1[i][j] = 0.2 * (A_local_1[i][j] + A_local_1[i][j-1] + A_local_1[i][1+j] + A_local_1[1+i][j] + A_local_1[i-1][j]);
	  } else {
	    B_local_2[i][j] = 0.2 * (A_local_2[i][j] + A_local_2[i][j-1] + A_local_2[i][1+j] + A_local_2[1+i][j] + A_local_2[i-1][j]);
	  }
	}
      for (i = 1; i < n - 1; i++)
	for (j = 1; j < n - 1; j++)
	{
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS DEPENDENCE variable=A_local_1 inter false
#pragma HLS DEPENDENCE variable=A_local_2 inter false
	  if (flag == 0) {
	    A_local_1[i][j] = 0.2 * (B_local_1[i][j] + B_local_1[i][j-1] + B_local_1[i][1+j] + B_local_1[1+i][j] + B_local_1[i-1][j]);
	  } else {
	    A_local_2[i][j] = 0.2 * (B_local_2[i][j] + B_local_2[i][j-1] + B_local_2[i][1+j] + B_local_2[1+i][j] + B_local_2[i-1][j]);
	  }
	}
    }
}

// Store phase: write the selected buffer set back to global memory
static void store(MARS_WIDE_BUS_TYPE *A, MARS_WIDE_BUS_TYPE *B,
                  double A_local_1[N][N], double B_local_1[N][N],
                  double A_local_2[N][N], double B_local_2[N][N],
                  int flag)
{
  const int n = N;
  if (flag == 0) {
    for (int i = 0; i < n; i++) {
#pragma HLS LOOP_TRIPCOUNT min=N max=N
      memcpy_wide_bus_write_float(A, &A_local_1[i][0], (long)i * n * sizeof(double), n * sizeof(double));
      memcpy_wide_bus_write_float(B, &B_local_1[i][0], (long)i * n * sizeof(double), n * sizeof(double));
    }
  } else {
    for (int i = 0; i < n; i++) {
#pragma HLS LOOP_TRIPCOUNT min=N max=N
      memcpy_wide_bus_write_float(A, &A_local_2[i][0], (long)i * n * sizeof(double), n * sizeof(double));
      memcpy_wide_bus_write_float(B, &B_local_2[i][0], (long)i * n * sizeof(double), n * sizeof(double));
    }
  }
}

void kernel_jacobi_2d(

			    MARS_WIDE_BUS_TYPE *A,
			    MARS_WIDE_BUS_TYPE *B)
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  // Two copies of each local buffer for ping-pong double buffering
  static double A_local_1[N][N];
  static double B_local_1[N][N];
  static double A_local_2[N][N];
  static double B_local_2[N][N];
  // Partition along the row dimension so that the stencil's vertical
  // neighbors (i-1, i, i+1) can be read in parallel within the pipeline.
#pragma HLS ARRAY_PARTITION variable=A_local_1 cyclic factor=3 dim=1
#pragma HLS ARRAY_PARTITION variable=B_local_1 cyclic factor=3 dim=1
#pragma HLS ARRAY_PARTITION variable=A_local_2 cyclic factor=3 dim=1
#pragma HLS ARRAY_PARTITION variable=B_local_2 cyclic factor=3 dim=1
  // Partition along the column dimension to feed the unrolled inner loop
  // so multiple horizontal neighbors can be accessed concurrently.
#pragma HLS ARRAY_PARTITION variable=A_local_1 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=B_local_1 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=A_local_2 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=B_local_2 cyclic factor=8 dim=2

  // With a single working set, the schedule is:
  //   load(set 0) -> compute(set 0) -> store(set 0)
  // The double-buffer structure (ping-pong via flag) lets the load of one
  // buffer set overlap with compute/store on the other when multiple
  // invocations or tiles are processed back-to-back.
  int flag = 0;

  // ---- LOAD into buffer set 'flag' ----
  load(A, B, A_local_1, B_local_1, A_local_2, B_local_2, flag);

  // ---- COMPUTE on buffer set 'flag' ----
  compute(A_local_1, B_local_1, A_local_2, B_local_2, flag);

  // ---- STORE from buffer set 'flag' ----
  store(A, B, A_local_1, B_local_1, A_local_2, B_local_2, flag);
}