#include "gemm.h"

#define TILE_J 256

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
    const int nk = NK;

  int i, j, k, jj;

  // Local tile buffers.
  double A_row[NK];        // staged row of A (full K dimension)
  double C_tile[TILE_J];   // staged C row tile
  double B_tile[NK][TILE_J]; // staged B sub-block for current j-tile

  // Partition B_tile and C_tile along the j dimension to match the unroll
  // factor of the compute_j loop so parallel iterations access distinct banks.
#pragma HLS ARRAY_PARTITION variable=B_tile cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=C_tile cyclic factor=4 dim=1

  for (i = 0; i < ni; i++) {

    // ---- LOAD A row (reused across all j-tiles) ----
  load_A:
    for (k = 0; k < nk; k++) {
#pragma HLS PIPELINE II=1
      A_row[k] = A[i][k];
    }

    // Process C row in tiles along j.
    for (jj = 0; jj < nj; jj += TILE_J) {
      int j_end = jj + TILE_J;
      if (j_end > nj) j_end = nj;
      int tj = j_end - jj;

      // ---- LOAD: stage C tile and apply beta scaling ----
    load_C:
      for (j = 0; j < tj; j++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_J
#pragma HLS PIPELINE II=1
        C_tile[j] = C[i][jj + j] * beta;
      }

      // ---- LOAD: stage B sub-block for this j-tile ----
    load_B:
      for (k = 0; k < nk; k++) {
        for (j = 0; j < tj; j++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_J
#pragma HLS PIPELINE II=1
          B_tile[k][j] = B[k][jj + j];
        }
      }

      // ---- COMPUTE: accumulate over k on local buffers ----
    compute_k:
      for (k = 0; k < nk; k++) {
        double a_val = alpha * A_row[k];
      compute_j:
        for (j = 0; j < tj; j++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_J
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
          // Each j-iteration touches a distinct C_tile[j] element, so the
          // accumulation has no real loop-carried dependence across j.
#pragma HLS DEPENDENCE variable=C_tile inter false
          C_tile[j] += a_val * B_tile[k][j];
        }
      }

      // ---- STORE: write back computed C tile ----
    store_C:
      for (j = 0; j < tj; j++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_J
#pragma HLS PIPELINE II=1
        C[i][jj + j] = C_tile[j];
      }
    }
  }

}