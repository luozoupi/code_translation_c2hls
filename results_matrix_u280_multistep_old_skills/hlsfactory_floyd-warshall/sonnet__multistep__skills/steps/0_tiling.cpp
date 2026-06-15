#include "floyd-warshall.h"

extern "C" {

static const int TILE = 16;

void load_k_row_tile(int path[N][N], int k, int j_start, int buf[TILE])
{
    for (int j = 0; j < TILE; j++) {
#pragma HLS PIPELINE II=1
        int jj = j_start + j;
        buf[j] = (jj < N) ? path[k][jj] : 0;
    }
}

void load_ik_col_tile(int path[N][N], int k, int i_start, int buf[TILE])
{
    for (int i = 0; i < TILE; i++) {
#pragma HLS PIPELINE II=1
        int ii = i_start + i;
        buf[i] = (ii < N) ? path[ii][k] : 0;
    }
}

void load_tile(int path[N][N], int i_start, int j_start, int tile[TILE][TILE])
{
    for (int i = 0; i < TILE; i++) {
#pragma HLS PIPELINE II=1
        for (int j = 0; j < TILE; j++) {
            int ii = i_start + i;
            int jj = j_start + j;
            tile[i][j] = (ii < N && jj < N) ? path[ii][jj] : 0;
        }
    }
}

void compute_tile(int tile[TILE][TILE], int ik_col[TILE], int kj_row[TILE])
{
    for (int i = 0; i < TILE; i++) {
#pragma HLS PIPELINE II=1
        for (int j = 0; j < TILE; j++) {
            int new_val = ik_col[i] + kj_row[j];
            int old_val = tile[i][j];
            tile[i][j] = (old_val < new_val) ? old_val : new_val;
        }
    }
}

void store_tile(int path[N][N], int i_start, int j_start, int tile[TILE][TILE])
{
    for (int i = 0; i < TILE; i++) {
#pragma HLS PIPELINE II=1
        for (int j = 0; j < TILE; j++) {
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
#pragma HLS INTERFACE m_axi port=path offset=slave bundle=gmem depth=32400
#pragma HLS INTERFACE s_axilite port=path bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local tile buffers
    int local_tile[TILE][TILE];
    int kj_row[TILE];   // row k, columns [j_start : j_start+TILE]
    int ik_col[TILE];   // column k, rows [i_start : i_start+TILE]

#pragma HLS ARRAY_PARTITION variable=local_tile complete dim=2
#pragma HLS ARRAY_PARTITION variable=kj_row complete dim=1
#pragma HLS ARRAY_PARTITION variable=ik_col complete dim=1

    for (int k = 0; k < N; k++) {
        // Tile over i and j
        for (int i_start = 0; i_start < N; i_start += TILE) {
            for (int j_start = 0; j_start < N; j_start += TILE) {

                // Load phase: bring in the tile and required row/col segments
                load_k_row_tile(path, k, j_start, kj_row);
                load_ik_col_tile(path, k, i_start, ik_col);
                load_tile(path, i_start, j_start, local_tile);

                // Compute phase: operate entirely on local buffers
                compute_tile(local_tile, ik_col, kj_row);

                // Store phase: write tile back to global memory
                store_tile(path, i_start, j_start, local_tile);
            }
        }
    }
}

} // extern "C"