#include "floyd-warshall.h"

extern "C" {

static const int TILE = 16;

void load_k_row_tile(int path[N][N], int k, int j_start, int buf[TILE])
{
#pragma HLS ARRAY_PARTITION variable=buf complete dim=1
    for (int j = 0; j < TILE; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
        int jj = j_start + j;
        buf[j] = (jj < N) ? path[k][jj] : 0;
    }
}

void load_ik_col_tile(int path[N][N], int k, int i_start, int buf[TILE])
{
#pragma HLS ARRAY_PARTITION variable=buf complete dim=1
    for (int i = 0; i < TILE; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
        int ii = i_start + i;
        buf[i] = (ii < N) ? path[ii][k] : 0;
    }
}

void load_tile(int path[N][N], int i_start, int j_start, int tile[TILE][TILE])
{
#pragma HLS ARRAY_PARTITION variable=tile complete dim=2
    for (int i = 0; i < TILE; i++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
        for (int j = 0; j < TILE; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
            int ii = i_start + i;
            int jj = j_start + j;
            tile[i][j] = (ii < N && jj < N) ? path[ii][jj] : 0;
        }
    }
}

void compute_tile(int tile[TILE][TILE], int ik_col[TILE], int kj_row[TILE])
{
#pragma HLS ARRAY_PARTITION variable=tile complete dim=2
#pragma HLS ARRAY_PARTITION variable=ik_col complete dim=1
#pragma HLS ARRAY_PARTITION variable=kj_row complete dim=1
    for (int i = 0; i < TILE; i++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
        for (int j = 0; j < TILE; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS DEPENDENCE variable=tile inter false
            int new_val = ik_col[i] + kj_row[j];
            int old_val = tile[i][j];
            tile[i][j] = (old_val < new_val) ? old_val : new_val;
        }
    }
}

void store_tile(int path[N][N], int i_start, int j_start, int tile[TILE][TILE])
{
#pragma HLS ARRAY_PARTITION variable=tile complete dim=2
    for (int i = 0; i < TILE; i++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
        for (int j = 0; j < TILE; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
            int ii = i_start + i;
            int jj = j_start + j;
            if (ii < N && jj < N) {
                path[ii][jj] = tile[i][j];
            }
        }
    }
}

void kernel_floyd_warshall(
    int path[N + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=path offset=slave bundle=gmem depth=32400 \
    max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE s_axilite port=path bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Double-buffered local tile buffers (ping-pong)
    int local_tile_0[TILE][TILE];
    int local_tile_1[TILE][TILE];
    int kj_row_0[TILE];
    int kj_row_1[TILE];
    int ik_col_0[TILE];
    int ik_col_1[TILE];

#pragma HLS ARRAY_PARTITION variable=local_tile_0 complete dim=2
#pragma HLS ARRAY_PARTITION variable=local_tile_1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=kj_row_0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=kj_row_1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=ik_col_0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=ik_col_1 complete dim=1

    // Total number of tiles per k iteration
    static const int N_TILES_I = (N + TILE - 1) / TILE;  // 12
    static const int N_TILES_J = (N + TILE - 1) / TILE;  // 12
    static const int TOTAL_TILES = N_TILES_I * N_TILES_J; // 144

    for (int k = 0; k < N; k++) {
#pragma HLS LOOP_TRIPCOUNT min=180 max=180

        // Pre-load the first tile (tile 0) into buffer set 0
        {
            int i_start = 0;
            int j_start = 0;
            load_k_row_tile(path, k, j_start, kj_row_0);
            load_ik_col_tile(path, k, i_start, ik_col_0);
            load_tile(path, i_start, j_start, local_tile_0);
        }

        for (int t = 0; t < TOTAL_TILES; t++) {
#pragma HLS LOOP_TRIPCOUNT min=144 max=144

            int i_tile = t / N_TILES_J;
            int j_tile = t % N_TILES_J;
            int i_start = i_tile * TILE;
            int j_start = j_tile * TILE;

            // Determine which buffer set is "current" (to compute) and "next" (to load)
            int ping = t % 2;  // 0 or 1

            // Compute next tile indices
            int t_next = t + 1;
            int i_tile_next = t_next / N_TILES_J;
            int j_tile_next = t_next % N_TILES_J;
            int i_start_next = i_tile_next * TILE;
            int j_start_next = j_tile_next * TILE;

            if (ping == 0) {
                // Compute using buffer set 0
                compute_tile(local_tile_0, ik_col_0, kj_row_0);

                // Store buffer set 0 result
                store_tile(path, i_start, j_start, local_tile_0);

                // Load next tile into buffer set 1 (if there is a next tile)
                if (t_next < TOTAL_TILES) {
                    load_k_row_tile(path, k, j_start_next, kj_row_1);
                    load_ik_col_tile(path, k, i_start_next, ik_col_1);
                    load_tile(path, i_start_next, j_start_next, local_tile_1);
                }
            } else {
                // Compute using buffer set 1
                compute_tile(local_tile_1, ik_col_1, kj_row_1);

                // Store buffer set 1 result
                store_tile(path, i_start, j_start, local_tile_1);

                // Load next tile into buffer set 0 (if there is a next tile)
                if (t_next < TOTAL_TILES) {
                    load_k_row_tile(path, k, j_start_next, kj_row_0);
                    load_ik_col_tile(path, k, i_start_next, ik_col_0);
                    load_tile(path, i_start_next, j_start_next, local_tile_0);
                }
            }
        }
    }
}

} // extern "C"