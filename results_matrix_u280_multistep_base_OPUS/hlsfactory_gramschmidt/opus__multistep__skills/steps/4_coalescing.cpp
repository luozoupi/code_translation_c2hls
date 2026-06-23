#include "gramschmidt.h"
#include <cstring>

#ifndef LARGE_BUS
#define LARGE_BUS 512
#endif

// Number of double (64-bit) elements per 512-bit wide word
#define WIDE_ELEMS (LARGE_BUS / 64)

// Wide bus word: 8 doubles = 512 bits. Plain POD struct so it compiles
// with a standard C++ compiler and still maps to a wide AXI port in HLS.
struct MARS_WIDE_BUS_TYPE {
  double data[WIDE_ELEMS];
};

// Read one double element at logical index `idx` from wide bus array.
static double memcpy_wide_bus_read_float(MARS_WIDE_BUS_TYPE *bus, long idx)
{
#pragma HLS INLINE
  long word = idx / WIDE_ELEMS;
  int off = (int)(idx % WIDE_ELEMS);
  return bus[word].data[off];
}

// Write one double element `val` at logical index `idx` to wide bus array.
static void memcpy_wide_bus_write_float(MARS_WIDE_BUS_TYPE *bus, double val, long idx)
{
#pragma HLS INLINE
  long word = idx / WIDE_ELEMS;
  int off = (int)(idx % WIDE_ELEMS);
  bus[word].data[off] = val;
}


// Load column j of A into buffer selected by flag
// A is stored row-major; column access reads A[i*N + j].
static void load_aj(MARS_WIDE_BUS_TYPE *A,
                    double a_j_1[M], double a_j_2[M],
                    int j, bool flag)
{
#pragma HLS INLINE off
  const int m = M;
load_aj:
  for (int i = 0; i < m; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=60 max=60
    double v = memcpy_wide_bus_read_float(A, (long)i * N + j);
    if (flag) a_j_1[i] = v; else a_j_2[i] = v;
  }
}

// Compute (dot, update) and store column j using buffer selected by flag
static void compute_store_aj(MARS_WIDE_BUS_TYPE *A,
                             MARS_WIDE_BUS_TYPE *R,
                             double q_col[M],
                             double a_j_1[M], double a_j_2[M],
                             int k, int j, bool flag)
{
#pragma HLS INLINE off
  const int m = M;

  // ---- COMPUTE: dot product q_col . a_j ----
  double r_acc = 0.0;
dot_loop:
  for (int i = 0; i < m; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=60 max=60
    double v = flag ? a_j_1[i] : a_j_2[i];
    r_acc += q_col[i] * v;
  }
  memcpy_wide_bus_write_float(R, r_acc, (long)k * N + j);

  // ---- COMPUTE: update column j in local buffer ----
  double r_val = r_acc;
update_loop:
  for (int i = 0; i < m; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=60 max=60
    double v = flag ? a_j_1[i] : a_j_2[i];
    v = v - q_col[i] * r_val;
    if (flag) a_j_1[i] = v; else a_j_2[i] = v;
  }

  // ---- STORE: write back updated column j of A ----
store_aj:
  for (int i = 0; i < m; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=60 max=60
    double out = flag ? a_j_1[i] : a_j_2[i];
    memcpy_wide_bus_write_float(A, out, (long)i * N + j);
  }
}


void kernel_gramschmidt(
			MARS_WIDE_BUS_TYPE *A,
			MARS_WIDE_BUS_TYPE *R,
			MARS_WIDE_BUS_TYPE *Q)
{
#pragma HLS INLINE off

    const int m = M;
    const int n = N;

  int i, j, k;

  double nrm;

  // Local tile buffers for column-wise reuse
  double a_col[M];   // staged column k of A
  double q_col[M];   // staged column k of Q
  // Double-buffered working columns for j_loop
  double a_j_1[M];
  double a_j_2[M];
#pragma HLS ARRAY_PARTITION variable=a_col cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=q_col cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=a_j_1 cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=a_j_2 cyclic factor=4 dim=1

  for (k = 0; k < n; k++)
    {
      // ---- LOAD: stage column k of A into local buffer ----
    load_acol:
      for (i = 0; i < m; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=60 max=60
        a_col[i] = memcpy_wide_bus_read_float(A, (long)i * N + k);
      }

      // ---- COMPUTE: norm of column k from local buffer ----
      nrm = 0.0;
    nrm_loop:
      for (i = 0; i < m; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=60 max=60
        nrm += a_col[i] * a_col[i];
      }

      double r_kk = sqrt(nrm);
      memcpy_wide_bus_write_float(R, r_kk, (long)k * N + k);

      // ---- COMPUTE: normalize column k into local q_col ----
    q_loop:
      for (i = 0; i < m; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=60 max=60
        q_col[i] = a_col[i] / r_kk;
      }

      // ---- STORE: write back Q column k ----
    store_qcol:
      for (i = 0; i < m; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=60 max=60
        memcpy_wide_bus_write_float(Q, q_col[i], (long)i * N + k);
      }

      // ---- DOUBLE-BUFFERED j_loop ----
      // Overlap load of column (j+1) with compute/store of column j.
      int total = n - (k + 1);   // number of j iterations
      if (total > 0) {
        // Prologue: load first column into buffer 1 (flag=true)
        load_aj(A, a_j_1, a_j_2, k + 1, true);

      j_loop:
        for (int t = 0; t < total; t++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=80
          int j_cur = (k + 1) + t;
          bool flag_cur = ((t % 2) == 0); // buffer used by current compute

          // Prefetch next column into the OTHER buffer (overlaps compute)
          if (t + 1 < total) {
            int j_next = j_cur + 1;
            bool flag_next = !flag_cur;
            load_aj(A, a_j_1, a_j_2, j_next, flag_next);
          }

          // Compute + store current column from its buffer
          compute_store_aj(A, R, q_col, a_j_1, a_j_2, k, j_cur, flag_cur);
        }
      }
    }

}

extern "C" {
void workload(
			MARS_WIDE_BUS_TYPE *A,
			MARS_WIDE_BUS_TYPE *R,
			MARS_WIDE_BUS_TYPE *Q)
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=R offset=slave bundle=gmem1 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=Q offset=slave bundle=gmem2 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=R bundle=control
#pragma HLS INTERFACE s_axilite port=Q bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    kernel_gramschmidt(A, R, Q);
}
}