#include "gemm.h"

#define TILE_J 256

// ---- LOAD phase: stage A row, C tile (beta-scaled), and B sub-block ----
static void load(
    int i, int jj, int tj,
    double beta,
    double C[NI + 0][NJ + 0],
    double A[NI + 0][NK + 0],
    double B[NK + 0][NJ + 0],
    double A_row[NK],
    double C_tile[TILE_J],
    double B_tile[NK][TILE_J])
{
    const int nk = NK;
    int k, j;

  load_A:
    for (k = 0; k < nk; k++) {
#pragma HLS PIPELINE II=1
        A_row[k] = A[i][k];
    }

  load_C:
    for (j = 0; j < tj; j++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_J
#pragma HLS PIPELINE II=1
        C_tile[j] = C[i][jj + j] * beta;
    }

  load_B:
    for (k = 0; k < nk; k++) {
        for (j = 0; j < tj; j++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_J
#pragma HLS PIPELINE II=1
            B_tile[k][j] = B[k][jj + j];
        }
    }
}

// ---- COMPUTE + STORE phase: accumulate over k, then write back ----
static void compute(
    int i, int jj, int tj,
    double alpha,
    double C[NI + 0][NJ + 0],
    double A_row[NK],
    double C_tile[TILE_J],
    double B_tile[NK][TILE_J])
{
    const int nk = NK;
    int k, j;

  compute_k:
    for (k = 0; k < nk; k++) {
        double a_val = alpha * A_row[k];
      compute_j:
        for (j = 0; j < tj; j++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_J
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS DEPENDENCE variable=C_tile inter false
            C_tile[j] += a_val * B_tile[k][j];
        }
    }

  store_C:
    for (j = 0; j < tj; j++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_J
#pragma HLS PIPELINE II=1
        C[i][jj + j] = C_tile[j];
    }
}

void kernel_gemm(
		 double alpha,
		 double beta,
		 double C[ NI + 0][NJ + 0],
		 double A[ NI + 0][NK + 0],
		 double B[ NK + 0][NJ + 0])
{
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=alpha bundle=control
#pragma HLS INTERFACE s_axilite port=beta bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int ni = NI;
    const int nj = NJ;

  int i, jj;

  // ---- DOUBLE-BUFFERED local tile buffers (ping-pong pair) ----
  double A_row_1[NK];
  double A_row_2[NK];
  double C_tile_1[TILE_J];
  double C_tile_2[TILE_J];
  double B_tile_1[NK][TILE_J];
  double B_tile_2[NK][TILE_J];

  // Partition both buffer sets along the j dimension to match the unroll
  // factor of the compute_j loop so parallel iterations access distinct banks.
#pragma HLS ARRAY_PARTITION variable=B_tile_1 cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=B_tile_2 cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=C_tile_1 cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=C_tile_2 cyclic factor=4 dim=1

  // Flatten the (i, jj) iteration space into a single tile sequence so that
  // load of tile (t+1) can overlap compute/store of tile (t).
  // Since TILE_J >= NJ here, there is one j-tile per row, so the ping-pong
  // alternates across the row (i) dimension.

  // We pipeline the outer tile loop by software-pipelining the phases:
  //   - load tile 0 into buffer set 0
  //   - then for each tile t: compute(t in set b) while loading t+1 into set !b
  //   - finally compute the last loaded tile

  // Build list of (i, jj, tj) tiles implicitly via nested loops; for the
  // common case nj <= TILE_J there is exactly one jj iteration.

  // Determine number of j-tiles per row.
  // For generality we iterate (i, jj) with an explicit flag toggle.

  // ---- Prologue: load the first tile ----
  int first_i = 0;
  int first_jj = 0;
  int first_tj = (nj < TILE_J) ? nj : TILE_J;
  load(first_i, first_jj, first_tj, beta, C, A, B,
       A_row_1, C_tile_1, B_tile_1);

  int flag = 0;          // which buffer set currently holds loaded data
  int cur_i = first_i;
  int cur_jj = first_jj;
  int cur_tj = first_tj;

  // ---- Steady state: overlap load(next) with compute(current) ----
  for (i = 0; i < ni; i++) {
    for (jj = 0; jj < nj; jj += TILE_J) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=NI

      // Skip the very first tile (already prologue-loaded as current).
      if (i == 0 && jj == 0) continue;

      int j_end = jj + TILE_J;
      if (j_end > nj) j_end = nj;
      int tj = j_end - jj;

      // Load the next tile into the OTHER buffer set while computing current.
      if (flag == 0) {
        load(i, jj, tj, beta, C, A, B,
             A_row_2, C_tile_2, B_tile_2);
        compute(cur_i, cur_jj, cur_tj, alpha, C,
                A_row_1, C_tile_1, B_tile_1);
      } else {
        load(i, jj, tj, beta, C, A, B,
             A_row_1, C_tile_1, B_tile_1);
        compute(cur_i, cur_jj, cur_tj, alpha, C,
                A_row_2, C_tile_2, B_tile_2);
      }

      // Advance: the freshly loaded tile becomes the current tile.
      flag = 1 - flag;
      cur_i = i;
      cur_jj = jj;
      cur_tj = tj;
    }
  }

  // ---- Epilogue: compute the last loaded tile ----
  if (flag == 0) {
    compute(cur_i, cur_jj, cur_tj, alpha, C,
            A_row_1, C_tile_1, B_tile_1);
  } else {
    compute(cur_i, cur_jj, cur_tj, alpha, C,
            A_row_2, C_tile_2, B_tile_2);
  }
}