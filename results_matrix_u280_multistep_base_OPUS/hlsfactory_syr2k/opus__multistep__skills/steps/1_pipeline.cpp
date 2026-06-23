#include "syr2k.h"

#define TILE 16

void kernel_syr2k( 
		  double alpha,
		  double beta,
		  double C[ N + 0][N + 0],
		  double A[ N + 0][M + 0],
		  double B[ N + 0][M + 0])
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

    const int n = N;
    const int m = M;

    // Stage the full A and B matrices once: every output row needs rows 0..i
    // of A and B, so the entire A/B working set is reused across tiles.
    double A_buf[N][M];
    double B_buf[N][M];
#pragma HLS ARRAY_PARTITION variable=A_buf cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=B_buf cyclic factor=8 dim=2

    // ---- Load A and B (full reuse working set) ----
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < m; k++) {
#pragma HLS PIPELINE II=1
            A_buf[i][k] = A[i][k];
            B_buf[i][k] = B[i][k];
        }
    }

    // ---- Process output rows in tiles of TILE rows ----
    for (int it = 0; it < n; it += TILE) {
        int i_end = (it + TILE < n) ? (it + TILE) : n;
        int trows = i_end - it;

        // Local tile buffer for the C rows being updated.
        double C_tile[TILE][N];
#pragma HLS ARRAY_PARTITION variable=C_tile cyclic factor=2 dim=1

        // ---- Load phase: bring C tile rows into local buffer ----
        for (int ti = 0; ti < trows; ti++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
            int i = it + ti;
            for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
                C_tile[ti][j] = C[i][j];
            }
        }

        // ---- Compute phase: operate on local buffers ----
        for (int ti = 0; ti < trows; ti++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
            int i = it + ti;

            // Scale lower-triangular part by beta.
            for (int j = 0; j <= i; j++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=N
#pragma HLS PIPELINE II=1
                C_tile[ti][j] *= beta;
            }

            // Accumulate the syr2k update.
            for (int j = 0; j <= i; j++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=N
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=C_tile inter false
                double acc = C_tile[ti][j];
                for (int k = 0; k < m; k++) {
#pragma HLS UNROLL
                    acc += A_buf[j][k] * alpha * B_buf[i][k]
                         + B_buf[j][k] * alpha * A_buf[i][k];
                }
                C_tile[ti][j] = acc;
            }
        }

        // ---- Store phase: write tile back to global memory ----
        for (int ti = 0; ti < trows; ti++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
            int i = it + ti;
            for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
                C[i][j] = C_tile[ti][j];
            }
        }
    }
}