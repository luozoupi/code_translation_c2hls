#include "gesummv.h"

#define TILE 16  // tile size: process TILE rows at a time, TILE columns at a time

extern "C" {

void kernel_gesummv(
		    double alpha,
		    double beta,
		    double A[ N + 0][N + 0],
		    double B[ N + 0][N + 0],
		    double tmp[ N + 0],
		    double x[ N + 0],
		    double y[ N + 0])
{
#pragma HLS INTERFACE m_axi port=A   offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=B   offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=tmp offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=x   offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=y   offset=slave bundle=gmem4
#pragma HLS INTERFACE s_axilite port=alpha  bundle=control
#pragma HLS INTERFACE s_axilite port=beta   bundle=control
#pragma HLS INTERFACE s_axilite port=A      bundle=control
#pragma HLS INTERFACE s_axilite port=B      bundle=control
#pragma HLS INTERFACE s_axilite port=tmp    bundle=control
#pragma HLS INTERFACE s_axilite port=x      bundle=control
#pragma HLS INTERFACE s_axilite port=y      bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local tile buffers for A and B rows, and a tile of x
    double A_local[TILE][TILE];
    double B_local[TILE][TILE];
    double x_local[TILE];

    // Partition dim=1 (row) completely so each row is independently accessible,
    // and partition dim=2 (col) completely for parallel access in pipelined k-loop
#pragma HLS ARRAY_PARTITION variable=A_local complete dim=1
#pragma HLS ARRAY_PARTITION variable=A_local complete dim=2
#pragma HLS ARRAY_PARTITION variable=B_local complete dim=1
#pragma HLS ARRAY_PARTITION variable=B_local complete dim=2
#pragma HLS ARRAY_PARTITION variable=x_local complete dim=1

    // Accumulators for a tile of output rows
    double tmp_acc[TILE];
    double y_acc[TILE];

#pragma HLS ARRAY_PARTITION variable=tmp_acc complete dim=1
#pragma HLS ARRAY_PARTITION variable=y_acc   complete dim=1

    // Tile over output rows (i dimension)
    for (int row_start = 0; row_start < N; row_start += TILE) {
#pragma HLS LOOP_TRIPCOUNT min=6 max=6
        int row_tile = ((row_start + TILE) <= N) ? TILE : (N - row_start);

        // Initialize accumulators for this row tile
        init_acc: for (int i = 0; i < TILE; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
            tmp_acc[i] = 0.0;
            y_acc[i]   = 0.0;
        }

        // Tile over columns (k dimension) — x is reused across all row tiles
        for (int col_start = 0; col_start < N; col_start += TILE) {
#pragma HLS LOOP_TRIPCOUNT min=6 max=6
            int col_tile = ((col_start + TILE) <= N) ? TILE : (N - col_start);

            // --- LOAD PHASE ---
            // Load a tile of rows from A and B: pipeline inner k loop
            load_AB: for (int i = 0; i < row_tile; i++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
                for (int k = 0; k < col_tile; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
                    A_local[i][k] = A[row_start + i][col_start + k];
                    B_local[i][k] = B[row_start + i][col_start + k];
                }
            }

            // Load tile of x
            load_x: for (int k = 0; k < col_tile; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
                x_local[k] = x[col_start + k];
            }

            // --- COMPUTE PHASE ---
            // Pipeline the inner k loop; outer i loop iterates over rows
            // tmp_acc[i] and y_acc[i] have no cross-iteration dependency on k
            compute_tile: for (int i = 0; i < row_tile; i++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
                for (int k = 0; k < col_tile; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS DEPENDENCE variable=tmp_acc inter false
#pragma HLS DEPENDENCE variable=y_acc   inter false
#pragma HLS DEPENDENCE variable=A_local inter false
#pragma HLS DEPENDENCE variable=B_local inter false
#pragma HLS DEPENDENCE variable=x_local inter false
                    tmp_acc[i] += A_local[i][k] * x_local[k];
                    y_acc[i]   += B_local[i][k] * x_local[k];
                }
            }
        }

        // --- STORE PHASE ---
        // Write completed row tile results back to global memory
        store_out: for (int i = 0; i < row_tile; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
            tmp[row_start + i] = tmp_acc[i];
            y[row_start + i]   = alpha * tmp_acc[i] + beta * y_acc[i];
        }
    }
}

} // extern "C"