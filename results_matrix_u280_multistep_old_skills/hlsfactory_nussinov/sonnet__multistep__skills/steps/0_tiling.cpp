#include "nussinov.h"

#define TILE_SIZE 16

extern "C" {

void kernel_nussinov(char seq[N + 0],
                     int table[N + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=seq   offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=table offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=seq    bundle=control
#pragma HLS INTERFACE s_axilite port=table  bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local buffers
    char  l_seq[N];
    int   l_table[N][N];

#pragma HLS ARRAY_PARTITION variable=l_seq   complete dim=1
#pragma HLS ARRAY_PARTITION variable=l_table cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_table cyclic factor=8 dim=2

    const int n = N;

    // =========================================================
    // LOAD PHASE: load seq in tiles
    // =========================================================
    load_seq_tiles: for (int t = 0; t < n; t += TILE_SIZE) {
        int tile_end = (t + TILE_SIZE < n) ? (t + TILE_SIZE) : n;
        load_seq_inner: for (int ii = t; ii < tile_end; ii++) {
#pragma HLS PIPELINE II=1
            l_seq[ii] = seq[ii];
        }
    }

    // =========================================================
    // LOAD PHASE: load table in row-tiles
    // =========================================================
    load_table_row_tile: for (int ti = 0; ti < n; ti += TILE_SIZE) {
        int ti_end = (ti + TILE_SIZE < n) ? (ti + TILE_SIZE) : n;
        load_table_col_tile: for (int tj = 0; tj < n; tj += TILE_SIZE) {
            int tj_end = (tj + TILE_SIZE < n) ? (tj + TILE_SIZE) : n;
            load_table_i: for (int ii = ti; ii < ti_end; ii++) {
                load_table_j: for (int jj = tj; jj < tj_end; jj++) {
#pragma HLS PIPELINE II=1
                    l_table[ii][jj] = table[ii][jj];
                }
            }
        }
    }

    // =========================================================
    // COMPUTE PHASE: Nussinov DP with tiled inner k-loop
    // =========================================================
    int i, j, k;

    for (i = n - 1; i >= 0; i--) {
        for (j = i + 1; j < n; j++) {

            if (j - 1 >= 0)
                l_table[i][j] = ((l_table[i][j] >= l_table[i][j-1]) ? l_table[i][j] : l_table[i][j-1]);
            if (i + 1 < n)
                l_table[i][j] = ((l_table[i][j] >= l_table[i+1][j]) ? l_table[i][j] : l_table[i+1][j]);

            if (j - 1 >= 0 && i + 1 < n) {
                if (i < j - 1)
                    l_table[i][j] = ((l_table[i][j] >= l_table[i+1][j-1] + (((l_seq[i]) + (l_seq[j])) == 3 ? 1 : 0)) ? l_table[i][j] : l_table[i+1][j-1] + (((l_seq[i]) + (l_seq[j])) == 3 ? 1 : 0));
                else
                    l_table[i][j] = ((l_table[i][j] >= l_table[i+1][j-1]) ? l_table[i][j] : l_table[i+1][j-1]);
            }

            // Tiled inner k-loop: process k in tiles of TILE_SIZE
            int tij = l_table[i][j];

            compute_k_tile: for (int kt = i + 1; kt < j; kt += TILE_SIZE) {
                int kt_end = (kt + TILE_SIZE < j) ? (kt + TILE_SIZE) : j;

                // Load tile of l_table[i][kt..kt_end-1] and l_table[kt+1..kt_end][j]
                // into local tile buffers
                int tile_row[TILE_SIZE];  // l_table[i][k]
                int tile_col[TILE_SIZE];  // l_table[k+1][j]
#pragma HLS ARRAY_PARTITION variable=tile_row complete dim=1
#pragma HLS ARRAY_PARTITION variable=tile_col complete dim=1

                // Load tile from local buffer (staged access)
                load_tile_k: for (int kk = 0; kk < TILE_SIZE; kk++) {
#pragma HLS PIPELINE II=1
                    int k_idx = kt + kk;
                    if (k_idx < kt_end) {
                        tile_row[kk] = l_table[i][k_idx];
                        tile_col[kk] = l_table[k_idx + 1][j];
                    } else {
                        tile_row[kk] = 0;
                        tile_col[kk] = 0;
                    }
                }

                // Compute over tile
                compute_tile_k: for (int kk = 0; kk < TILE_SIZE; kk++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
                    int k_idx = kt + kk;
                    if (k_idx < kt_end) {
                        int val = tile_row[kk] + tile_col[kk];
                        tij = (tij >= val) ? tij : val;
                    }
                }
            }
            l_table[i][j] = tij;
        }
    }

    // =========================================================
    // STORE PHASE: store table back in row-tiles
    // =========================================================
    store_table_row_tile: for (int ti = 0; ti < n; ti += TILE_SIZE) {
        int ti_end = (ti + TILE_SIZE < n) ? (ti + TILE_SIZE) : n;
        store_table_col_tile: for (int tj = 0; tj < n; tj += TILE_SIZE) {
            int tj_end = (tj + TILE_SIZE < n) ? (tj + TILE_SIZE) : n;
            store_table_i: for (int ii = ti; ii < ti_end; ii++) {
                store_table_j: for (int jj = tj; jj < tj_end; jj++) {
#pragma HLS PIPELINE II=1
                    table[ii][jj] = l_table[ii][jj];
                }
            }
        }
    }
}

} // extern "C"