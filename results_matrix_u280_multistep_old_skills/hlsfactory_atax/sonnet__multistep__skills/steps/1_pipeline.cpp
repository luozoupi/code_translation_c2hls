#include "atax.h"

// Tile size for rows of A
#define TILE_M 16

static void load_x_buf(double x[N], double l_x[N]) {
    load_x: for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
        l_x[j] = x[j];
    }
}

static void load_A_tile(double A[M][N], double l_A[TILE_M][N],
                        int row_start, int rows) {
    load_A: for (int i = 0; i < rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_M
        for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=N max=N
            l_A[i][j] = A[row_start + i][j];
        }
    }
}

static void compute_tile(double l_A[TILE_M][N], double l_x[N],
                         double l_tmp_tile[TILE_M], double l_y[N],
                         int rows) {
#pragma HLS ARRAY_PARTITION variable=l_A        cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_x        cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_y        cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_tmp_tile complete dim=1

    compute_rows: for (int i = 0; i < rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_M
        double tmp_i = 0.0;

        // Compute tmp[i] = sum_j A[i][j] * x[j]
        loop_tmp: for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=N max=N
#pragma HLS DEPENDENCE variable=l_A inter false
#pragma HLS DEPENDENCE variable=l_x inter false
            tmp_i += l_A[i][j] * l_x[j];
        }
        l_tmp_tile[i] = tmp_i;

        // Accumulate y[j] += A[i][j] * tmp[i]
        loop_y: for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=N max=N
#pragma HLS DEPENDENCE variable=l_A inter false
#pragma HLS DEPENDENCE variable=l_y inter false
            l_y[j] += l_A[i][j] * tmp_i;
        }
    }
}

static void store_tmp_tile(double tmp[M], double l_tmp_tile[TILE_M],
                           int row_start, int rows) {
    store_tmp: for (int i = 0; i < rows; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_M
        tmp[row_start + i] = l_tmp_tile[i];
    }
}

static void store_y_buf(double y[N], double l_y[N]) {
    store_y: for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
        y[j] = l_y[j];
    }
}

void kernel_atax(
    double A[M + 0][N + 0],
    double x[N + 0],
    double y[N + 0],
    double tmp[M + 0])
{
#pragma HLS INTERFACE m_axi port=A   offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=x   offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=y   offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=tmp offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=A      bundle=control
#pragma HLS INTERFACE s_axilite port=x      bundle=control
#pragma HLS INTERFACE s_axilite port=y      bundle=control
#pragma HLS INTERFACE s_axilite port=tmp    bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local buffers
    double l_x[N];
    double l_y[N];
    // Tile buffers — only TILE_M rows at a time instead of all M rows
    double l_A[TILE_M][N];
    double l_tmp_tile[TILE_M];

#pragma HLS ARRAY_PARTITION variable=l_x        cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_y        cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_tmp_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=l_A        cyclic factor=8 dim=2

    // Phase 1: Load x into local buffer (reused across all tiles)
    load_x_buf(x, l_x);

    // Phase 2: Initialize y accumulator
    init_y: for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
        l_y[j] = 0.0;
    }

    // Phase 3: Process A in row-tiles
    tile_loop: for (int row_start = 0; row_start < M; row_start += TILE_M) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=(M+TILE_M-1)/TILE_M
        // Number of rows in this tile (handle boundary)
        int rows = (row_start + TILE_M <= M) ? TILE_M : (M - row_start);

        // Load: bring current tile of A into local buffer
        load_A_tile(A, l_A, row_start, rows);

        // Compute: work entirely on local buffers
        compute_tile(l_A, l_x, l_tmp_tile, l_y, rows);

        // Store: write tmp results for this tile back to global memory
        store_tmp_tile(tmp, l_tmp_tile, row_start, rows);
    }

    // Phase 4: Store accumulated y back to global memory
    store_y_buf(y, l_y);
}