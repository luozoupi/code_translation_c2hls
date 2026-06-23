#include "jacobi-2d.h"
#include <cstring>

// Load phase: stage global data into the selected buffer set
static void load(double A[N][N], double B[N][N],
                 double A_local_1[N][N], double B_local_1[N][N],
                 double A_local_2[N][N], double B_local_2[N][N],
                 int flag)
{
  const int n = N;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
    {
#pragma HLS PIPELINE II=1
      if (flag == 0) {
        A_local_1[i][j] = A[i][j];
        B_local_1[i][j] = B[i][j];
      } else {
        A_local_2[i][j] = A[i][j];
        B_local_2[i][j] = B[i][j];
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
static void store(double A[N][N], double B[N][N],
                  double A_local_1[N][N], double B_local_1[N][N],
                  double A_local_2[N][N], double B_local_2[N][N],
                  int flag)
{
  const int n = N;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
    {
#pragma HLS PIPELINE II=1
      if (flag == 0) {
        A[i][j] = A_local_1[i][j];
        B[i][j] = B_local_1[i][j];
      } else {
        A[i][j] = A_local_2[i][j];
        B[i][j] = B_local_2[i][j];
      }
    }
}

void kernel_jacobi_2d(

			    double A[ N + 0][N + 0],
			    double B[ N + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1
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