#include "cholesky.h"
#include <cstring>

// ---- Wide bus definitions (POD-based, no ap_int dependency) ----
#ifndef LARGE_BUS
#define LARGE_BUS 512
#endif

// number of doubles packed per wide bus word (512 / 64 = 8)
#define DOUBLES_PER_BUS (LARGE_BUS / 64)

// 512-bit wide bus word represented as a packed struct of doubles.
typedef struct {
  double lane[DOUBLES_PER_BUS];
} MARS_WIDE_BUS_TYPE;

// Wide bus read: copy 'num_bytes' bytes from global wide-bus memory (at byte
// offset 'offset_bytes') into a local double buffer.
static void memcpy_wide_bus_read_double(double *local, MARS_WIDE_BUS_TYPE *bus,
                                        long offset_bytes, long num_bytes)
{
#pragma HLS INLINE off
  long num_doubles = num_bytes / (long)sizeof(double);
  long start_elem = offset_bytes / (long)sizeof(double);
  long end_elem = start_elem + num_doubles;
  long word0 = start_elem / DOUBLES_PER_BUS;
  long wordE = (end_elem + DOUBLES_PER_BUS - 1) / DOUBLES_PER_BUS;

  read_outer:
  for (long w = word0; w < wordE; w++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=N
    MARS_WIDE_BUS_TYPE val = bus[w];
    read_inner:
    for (int s = 0; s < DOUBLES_PER_BUS; s++) {
#pragma HLS PIPELINE II=1
      long pos = w * DOUBLES_PER_BUS + s;
      if (pos >= start_elem && pos < end_elem) {
        local[pos - start_elem] = val.lane[s];
      }
    }
  }
}

// Wide bus write: copy 'num_bytes' bytes from a local double buffer into global
// wide-bus memory (at byte offset 'offset_bytes'). Performs read-modify-write
// per word so partial words preserve unrelated lanes.
static void memcpy_wide_bus_write_double(MARS_WIDE_BUS_TYPE *bus, double *local,
                                         long offset_bytes, long num_bytes)
{
#pragma HLS INLINE off
  long num_doubles = num_bytes / (long)sizeof(double);
  long start_elem = offset_bytes / (long)sizeof(double);
  long end_elem = start_elem + num_doubles;
  long word0 = start_elem / DOUBLES_PER_BUS;
  long wordE = (end_elem + DOUBLES_PER_BUS - 1) / DOUBLES_PER_BUS;

  write_outer:
  for (long w = word0; w < wordE; w++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=N
    MARS_WIDE_BUS_TYPE val = bus[w];  // read-modify-write to keep unrelated lanes
    write_inner:
    for (int s = 0; s < DOUBLES_PER_BUS; s++) {
#pragma HLS PIPELINE II=1
      long pos = w * DOUBLES_PER_BUS + s;
      if (pos >= start_elem && pos < end_elem) {
        val.lane[s] = local[pos - start_elem];
      }
    }
    bus[w] = val;
  }
}

// Load row j into the selected buffer (using wide bus reads)
static void load_row_j(MARS_WIDE_BUS_TYPE *A, double row_j_1[N], double row_j_2[N], int j, int n, int flag)
{
#pragma HLS INLINE off
  // base offset (in doubles) of row j in the flattened A
  int base = j * n;
  if (flag == 0) {
    memcpy_wide_bus_read_double(row_j_1, A, (long)base * sizeof(double), (long)n * sizeof(double));
  } else {
    memcpy_wide_bus_read_double(row_j_2, A, (long)base * sizeof(double), (long)n * sizeof(double));
  }
}

// Compute off-diagonal entry using the selected buffer
static double compute_offdiag(double row_i[N], double row_j_1[N], double row_j_2[N], int j, int flag)
{
#pragma HLS INLINE off
  double acc = row_i[j];
  if (flag == 0) {
    comp_j0:
    for (int k = 0; k < j; k++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=N
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
      acc -= row_i[k] * row_j_1[k];
    }
    return acc / row_j_1[j];
  } else {
    comp_j1:
    for (int k = 0; k < j; k++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=N
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
      acc -= row_i[k] * row_j_2[k];
    }
    return acc / row_j_2[j];
  }
}

extern "C" {
void kernel_cholesky(
		     double A[ N + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem \
    max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;

  // Reinterpret the flat 2D array as a wide-bus pointer for coalesced access.
  MARS_WIDE_BUS_TYPE *Abus = (MARS_WIDE_BUS_TYPE *)A;

  int i, j, k;

  // Local tile buffers for the row being processed and a helper row.
  double row_i[N];
  // Double-buffered helper rows (ping-pong).
  double row_j_1[N];
  double row_j_2[N];
#pragma HLS ARRAY_PARTITION variable=row_i cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=row_j_1 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=row_j_2 cyclic factor=8 dim=1

  for (i = 0; i < n; i++) {

    // ---- LOAD phase: stage row i into local buffer (wide bus read) ----
    {
      int base = i * n;
      memcpy_wide_bus_read_double(row_i, Abus, (long)base * sizeof(double), (long)n * sizeof(double));
    }

    // ---- COMPUTE phase (off-diagonal entries) with DOUBLE BUFFERING ----
    if (i > 0) {
      // Prologue: preload row j=0 into buffer 1
      load_row_j(Abus, row_j_1, row_j_2, 0, n, 0);

      for (j = 0; j < i; j++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=N
        int flag = j % 2;             // buffer holding current row j
        int next_flag = (j + 1) % 2;  // buffer to load row j+1 into

        // Load next row (j+1) into the other buffer while we compute on flag.
        if (j + 1 < i) {
          load_row_j(Abus, row_j_1, row_j_2, j + 1, n, next_flag);
        }

        // Compute off-diagonal using current buffer.
        double val = compute_offdiag(row_i, row_j_1, row_j_2, j, flag);
        row_i[j] = val;   // keep local copy consistent for later k reads
      }
    }

    // ---- COMPUTE phase (diagonal entry) ----
    double diag = row_i[i];
    compute_diag:
    for (k = 0; k < i; k++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=N
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
      diag -= row_i[k] * row_i[k];
    }
    row_i[i] = sqrt(diag);

    // ---- STORE phase: write back updated row i (wide bus write) ----
    {
      int base = i * n;
      // write the first (i+1) elements of row i
      memcpy_wide_bus_write_double(Abus, row_i, (long)base * sizeof(double), (long)(i + 1) * sizeof(double));
    }
  }

}
}