#include "trmm.h"
#include <cstring>

// Tile size along the row dimension for double buffering.
#define TILE 10

// Load a tile of A rows and corresponding B rows into the selected buffer set.
static void load_tile(
        double A[M][M],
        double B[M][N],
        double A_buf[TILE][M],
        double B_buf[TILE][N],
        int row_base,
        int rows)
{
#pragma HLS INLINE off
    load_A_rows:
    for (int ii = 0; ii < rows; ii++) {
        load_A_cols:
        for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
            A_buf[ii][j] = A[row_base + ii][j];
        }
    }
    load_B_rows:
    for (int ii = 0; ii < rows; ii++) {
        load_B_cols:
        for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            B_buf[ii][j] = B[row_base + ii][j];
        }
    }
}

void kernel_trmm(
		 double alpha,
		 double A[ M + 0][M + 0],
		 double B[ M + 0][N + 0])
{
#pragma HLS INLINE off

    const int m = M;
    const int n = N;

    // ---- Stage the FULL local buffers first (needed because trmm reduction
    //      reads B_local[k][j] for all k > i, i.e. future rows) ----
    static double A_local[M][M];
    static double B_local[M][N];
#pragma HLS ARRAY_PARTITION variable=A_local cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=B_local cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=B_local cyclic factor=4 dim=2

    int i, j, k;

    // ---- DOUBLE-BUFFERED LOAD phase ----
    // Two ping-pong buffer sets for staging tiles of A and B rows.
    static double A_buf1[TILE][M];
    static double A_buf2[TILE][M];
    static double B_buf1[TILE][N];
    static double B_buf2[TILE][N];
#pragma HLS ARRAY_PARTITION variable=A_buf1 cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=A_buf2 cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=B_buf1 cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=B_buf2 cyclic factor=4 dim=2

    const int num_tiles = (m + TILE - 1) / TILE;

    // Prologue: load the first tile into buffer set 1.
    {
        int rows0 = (m < TILE) ? m : TILE;
        load_tile(A, B, A_buf1, B_buf1, 0, rows0);
    }

    // Main load loop: while committing tile t to the full local buffers,
    // load tile t+1 into the other ping-pong buffer set so load overlaps copy.
    load_tiles:
    for (int t = 0; t < num_tiles; t++) {
        int row_base = t * TILE;
        int rows = ((row_base + TILE) <= m) ? TILE : (m - row_base);

        // Flag selects which buffer set currently holds tile t.
        bool sel = (t % 2) == 0;

        // Prefetch next tile into the OTHER buffer set (overlaps the commit below).
        int next_base = (t + 1) * TILE;
        int next_rows = ((next_base + TILE) <= m) ? TILE : (m - next_base);
        if (t + 1 < num_tiles) {
            if (sel) {
                // tile t in buf1 -> load next into buf2
                load_tile(A, B, A_buf2, B_buf2, next_base, next_rows);
            } else {
                // tile t in buf2 -> load next into buf1
                load_tile(A, B, A_buf1, B_buf1, next_base, next_rows);
            }
        }

        // Commit current tile (from the selected buffer set) into full locals.
        commit_rows:
        for (int ii = 0; ii < rows; ii++) {
            commit_A_cols:
            for (j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
                A_local[row_base + ii][j] =
                    sel ? A_buf1[ii][j] : A_buf2[ii][j];
            }
            commit_B_cols:
            for (j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
                B_local[row_base + ii][j] =
                    sel ? B_buf1[ii][j] : B_buf2[ii][j];
            }
        }
    }

    // ---- COMPUTE phase: operate on full local buffers ----
    compute_i:
    for (i = 0; i < m; i++) {
        compute_j:
        for (j = 0; j < n; j++) {
#pragma HLS UNROLL factor=4
            double acc = B_local[i][j];
            compute_k:
            for (k = i + 1; k < m; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=0 max=M
                acc += A_local[k][i] * B_local[k][j];
            }
            B_local[i][j] = alpha * acc;
        }
    }

    // ---- STORE phase: write results back to global memory ----
    store_B_rows:
    for (i = 0; i < m; i++) {
        store_B_cols:
        for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            B[i][j] = B_local[i][j];
        }
    }
}

extern "C" {
void workload(
		 double alpha,
		 double A[ M + 0][M + 0],
		 double B[ M + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=alpha bundle=control
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    kernel_trmm(alpha, A, B);
}
}