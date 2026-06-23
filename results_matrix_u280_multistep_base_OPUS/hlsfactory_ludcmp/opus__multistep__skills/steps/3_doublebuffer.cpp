#include "ludcmp.h"
#include <cstring>

extern "C" {

// Load a tile of a matrix row into one of two ping-pong staging buffers.
static void load_tile(double A[N][N], double row_tile_1[256], double row_tile_2[256],
                      int i, int tj, int chunk, int flag)
{
    if (flag == 0) {
        memcpy(row_tile_1, &A[i][tj], chunk * sizeof(double));
    } else {
        memcpy(row_tile_2, &A[i][tj], chunk * sizeof(double));
    }
}

// Copy a staged tile (from one of two ping-pong buffers) into the local matrix.
static void compute_tile(double A_local[N][N], double row_tile_1[256], double row_tile_2[256],
                         int i, int tj, int chunk, int flag)
{
    if (flag == 0) {
        for (int j = 0; j < chunk; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
#pragma HLS UNROLL factor=8
            A_local[i][tj + j] = row_tile_1[j];
        }
    } else {
        for (int j = 0; j < chunk; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
#pragma HLS UNROLL factor=8
            A_local[i][tj + j] = row_tile_2[j];
        }
    }
}

void kernel_ludcmp(
		   double A[ N + 0][N + 0],
		   double b[ N + 0],
		   double x[ N + 0],
		   double y[ N + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=y offset=slave bundle=gmem3
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=b bundle=control
#pragma HLS INTERFACE s_axilite port=x bundle=control
#pragma HLS INTERFACE s_axilite port=y bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;
    const int TILE = 256;  // tile size for staging linear transfers

    int i, j, k;
    int t, tj;

    double w;

    // Local buffers to stage data from global memory for reuse during the
    // computation-heavy LU decomposition phase.
    static double A_local[N][N];
#pragma HLS ARRAY_PARTITION variable=A_local cyclic factor=8 dim=2
    static double b_local[N];
    static double x_local[N];
    static double y_local[N];
#pragma HLS ARRAY_PARTITION variable=y_local cyclic factor=4
#pragma HLS ARRAY_PARTITION variable=x_local cyclic factor=4

    // ---------------- LOAD PHASE (DOUBLE BUFFERED) ----------------
    // Two ping-pong staging buffers so the burst load of tile k+1 overlaps the
    // copy of tile k into A_local.
    static double row_tile_1[TILE];
    static double row_tile_2[TILE];
#pragma HLS ARRAY_PARTITION variable=row_tile_1 cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=row_tile_2 cyclic factor=8

    // Build a flattened iteration over (i, tj) tiles so we can pipeline the
    // load/compute alternation across consecutive tiles.
    {
        // Count of tiles per row.
        int tiles_per_row = (n + TILE - 1) / TILE;
        int total_tiles = n * tiles_per_row;

        // Helper lambdas replaced by inline index computation.
        // Prologue: load first tile.
        int idx = 0;
        int cur_i = 0;
        int cur_tj = 0;
        int cur_chunk = (cur_tj + TILE <= n) ? TILE : (n - cur_tj);
        load_tile(A, row_tile_1, row_tile_2, cur_i, cur_tj, cur_chunk, 0);

        // Steady state: for each tile, while computing the current tile, load
        // the next tile into the other buffer.
        for (idx = 0; idx < total_tiles; idx++) {
            int flag = idx % 2;

            // Current tile coordinates.
            int row = idx / tiles_per_row;
            int col_tile = idx % tiles_per_row;
            int tjj = col_tile * TILE;
            int chunk = (tjj + TILE <= n) ? TILE : (n - tjj);

            // Determine the next tile coordinates (if any).
            int nidx = idx + 1;
            int has_next = (nidx < total_tiles);
            int nrow = nidx / tiles_per_row;
            int ncol_tile = nidx % tiles_per_row;
            int ntjj = ncol_tile * TILE;
            int nchunk = (ntjj + TILE <= n) ? TILE : (n - ntjj);

            // Load next tile into the OTHER buffer while we compute current.
            if (has_next) {
                load_tile(A, row_tile_1, row_tile_2, nrow, ntjj, nchunk, 1 - flag);
            }
            // Compute (copy out) the current tile from its buffer.
            compute_tile(A_local, row_tile_1, row_tile_2, row, tjj, chunk, flag);
        }
    }

    // Load b from global memory into a local tile, then into b_local.
    for (t = 0; t < n; t += TILE) {
        int chunk = (t + TILE <= n) ? TILE : (n - t);
        double b_tile[TILE];
        memcpy(b_tile, &b[t], chunk * sizeof(double));
        for (j = 0; j < chunk; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
            b_local[t + j] = b_tile[j];
        }
    }

    // ---------------- COMPUTE PHASE ----------------
    // LU decomposition (in-place on local buffer).
    for (i = 0; i < n; i++) {
        for (j = 0; j < i; j++) {
            w = A_local[i][j];
            for (k = 0; k < j; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=0 max=120
#pragma HLS DEPENDENCE variable=A_local inter false
#pragma HLS UNROLL factor=4
                w -= A_local[i][k] * A_local[k][j];
            }
            A_local[i][j] = w / A_local[j][j];
        }
        for (j = i; j < n; j++) {
            w = A_local[i][j];
            for (k = 0; k < i; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=0 max=120
#pragma HLS DEPENDENCE variable=A_local inter false
#pragma HLS UNROLL factor=4
                w -= A_local[i][k] * A_local[k][j];
            }
            A_local[i][j] = w;
        }
    }

    // Forward substitution to compute y.
    for (i = 0; i < n; i++) {
        w = b_local[i];
        for (j = 0; j < i; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=0 max=120
#pragma HLS DEPENDENCE variable=y_local inter false
#pragma HLS UNROLL factor=4
            w -= A_local[i][j] * y_local[j];
        }
        y_local[i] = w;
    }

    // Backward substitution to compute x.
    for (i = n - 1; i >= 0; i--) {
        w = y_local[i];
        for (j = i + 1; j < n; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=0 max=120
#pragma HLS DEPENDENCE variable=x_local inter false
#pragma HLS UNROLL factor=4
            w -= A_local[i][j] * x_local[j];
        }
        x_local[i] = w / A_local[i][i];
    }

    // ---------------- STORE PHASE ----------------
    // Write A back to global memory using tiled bursts within each row.
    for (i = 0; i < n; i++) {
        for (tj = 0; tj < n; tj += TILE) {
            int chunk = (tj + TILE <= n) ? TILE : (n - tj);
            double row_tile[TILE];
#pragma HLS ARRAY_PARTITION variable=row_tile cyclic factor=8
            for (j = 0; j < chunk; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
#pragma HLS UNROLL factor=8
                row_tile[j] = A_local[i][tj + j];
            }
            memcpy(&A[i][tj], row_tile, chunk * sizeof(double));
        }
    }

    // Write x and y back to global memory using tiled bursts.
    for (t = 0; t < n; t += TILE) {
        int chunk = (t + TILE <= n) ? TILE : (n - t);
        double x_tile[TILE];
        double y_tile[TILE];
        for (j = 0; j < chunk; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
            x_tile[j] = x_local[t + j];
            y_tile[j] = y_local[t + j];
        }
        memcpy(&x[t], x_tile, chunk * sizeof(double));
        memcpy(&y[t], y_tile, chunk * sizeof(double));
    }
}
}