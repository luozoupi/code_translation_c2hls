#include "atax.h"

// Tile size for rows of A
#define TILE_M 16

static void load_x_buf(double x[N], double l_x[N]) {
    load_x: for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
        l_x[j] = x[j];
    }
}

// Double-buffered load: flag selects which buffer to fill
static void load_A_tile(double A[M][N],
                        double l_A_0[TILE_M][N],
                        double l_A_1[TILE_M][N],
                        int flag,
                        int row_start, int rows) {
    if (flag == 0) {
        load_A_0: for (int i = 0; i < rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_M
            for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=N max=N
                l_A_0[i][j] = A[row_start + i][j];
            }
        }
    } else {
        load_A_1: for (int i = 0; i < rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_M
            for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=N max=N
                l_A_1[i][j] = A[row_start + i][j];
            }
        }
    }
}

// Double-buffered compute: flag selects which buffer to read from
static void compute_tile(double l_A_0[TILE_M][N],
                         double l_A_1[TILE_M][N],
                         double l_x[N],
                         double l_tmp_tile_0[TILE_M],
                         double l_tmp_tile_1[TILE_M],
                         double l_y[N],
                         int flag,
                         int rows) {
#pragma HLS ARRAY_PARTITION variable=l_A_0        cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_A_1        cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_x          cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_y          cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_tmp_tile_0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=l_tmp_tile_1 complete dim=1

    if (flag == 0) {
        compute_rows_0: for (int i = 0; i < rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_M
            double tmp_i = 0.0;
            loop_tmp_0: for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=N max=N
#pragma HLS DEPENDENCE variable=l_A_0 inter false
#pragma HLS DEPENDENCE variable=l_x   inter false
                tmp_i += l_A_0[i][j] * l_x[j];
            }
            l_tmp_tile_0[i] = tmp_i;
            loop_y_0: for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=N max=N
#pragma HLS DEPENDENCE variable=l_A_0 inter false
#pragma HLS DEPENDENCE variable=l_y   inter false
                l_y[j] += l_A_0[i][j] * tmp_i;
            }
        }
    } else {
        compute_rows_1: for (int i = 0; i < rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_M
            double tmp_i = 0.0;
            loop_tmp_1: for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=N max=N
#pragma HLS DEPENDENCE variable=l_A_1 inter false
#pragma HLS DEPENDENCE variable=l_x   inter false
                tmp_i += l_A_1[i][j] * l_x[j];
            }
            l_tmp_tile_1[i] = tmp_i;
            loop_y_1: for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=N max=N
#pragma HLS DEPENDENCE variable=l_A_1 inter false
#pragma HLS DEPENDENCE variable=l_y   inter false
                l_y[j] += l_A_1[i][j] * tmp_i;
            }
        }
    }
}

static void store_tmp_tile(double tmp[M],
                           double l_tmp_tile_0[TILE_M],
                           double l_tmp_tile_1[TILE_M],
                           int flag,
                           int row_start, int rows) {
    if (flag == 0) {
        store_tmp_0: for (int i = 0; i < rows; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_M
            tmp[row_start + i] = l_tmp_tile_0[i];
        }
    } else {
        store_tmp_1: for (int i = 0; i < rows; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_M
            tmp[row_start + i] = l_tmp_tile_1[i];
        }
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

    // Local buffers for x and y (single copy — reused across all tiles)
    double l_x[N];
    double l_y[N];

    // Double-buffered tile buffers (ping-pong)
    double l_A_0[TILE_M][N];
    double l_A_1[TILE_M][N];
    double l_tmp_tile_0[TILE_M];
    double l_tmp_tile_1[TILE_M];

#pragma HLS ARRAY_PARTITION variable=l_x          cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_y          cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_A_0        cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_A_1        cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_tmp_tile_0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=l_tmp_tile_1 complete dim=1

    // Phase 1: Load x into local buffer (reused across all tiles)
    load_x_buf(x, l_x);

    // Phase 2: Initialize y accumulator
    init_y: for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
        l_y[j] = 0.0;
    }

    // Compute total number of tiles
    const int num_tiles = (M + TILE_M - 1) / TILE_M;

    // Phase 3: Double-buffered tile loop
    // Prologue: preload tile 0 into buffer 0
    {
        int rows0 = (TILE_M <= M) ? TILE_M : M;
        load_A_tile(A, l_A_0, l_A_1, 0, 0, rows0);
    }

    tile_loop: for (int t = 0; t < num_tiles; t++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=(M+TILE_M-1)/TILE_M

        int row_start = t * TILE_M;
        int rows      = (row_start + TILE_M <= M) ? TILE_M : (M - row_start);

        // Current buffer: flag = t % 2
        // Next buffer:    flag = (t+1) % 2
        int cur_flag  = t % 2;
        int next_flag = 1 - cur_flag;

        // Prefetch next tile into the alternate buffer (if it exists)
        if (t + 1 < num_tiles) {
            int next_row_start = (t + 1) * TILE_M;
            int next_rows      = (next_row_start + TILE_M <= M)
                                     ? TILE_M
                                     : (M - next_row_start);
            load_A_tile(A, l_A_0, l_A_1, next_flag,
                        next_row_start, next_rows);
        }

        // Compute on current buffer
        compute_tile(l_A_0, l_A_1, l_x,
                     l_tmp_tile_0, l_tmp_tile_1,
                     l_y, cur_flag, rows);

        // Store tmp results for current tile
        store_tmp_tile(tmp,
                       l_tmp_tile_0, l_tmp_tile_1,
                       cur_flag,
                       row_start, rows);
    }

    // Phase 4: Store accumulated y back to global memory
    store_y_buf(y, l_y);
}