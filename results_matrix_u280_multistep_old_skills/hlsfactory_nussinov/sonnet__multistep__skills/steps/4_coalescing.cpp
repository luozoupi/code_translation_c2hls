#include "nussinov.h"
#include <string.h>

#define TILE_SIZE 16
#define UNROLL_FACTOR 8

extern "C" {

void kernel_nussinov(char seq[N + 0],
                     int table[N + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=seq   offset=slave bundle=gmem0 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=table offset=slave bundle=gmem1 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE s_axilite port=seq    bundle=control
#pragma HLS INTERFACE s_axilite port=table  bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local buffers
    char  l_seq[N];

    // Double-buffered table: two full copies for ping-pong
    int   l_table_0[N][N];
    int   l_table_1[N][N];

#pragma HLS ARRAY_PARTITION variable=l_seq     complete dim=1
#pragma HLS ARRAY_PARTITION variable=l_table_0 cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=l_table_0 cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=l_table_1 cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=l_table_1 cyclic factor=16 dim=2

    const int n = N;

    // =========================================================
    // LOAD PHASE: load seq using burst-friendly memcpy
    // =========================================================
    memcpy(l_seq, seq, sizeof(char) * n);

    // =========================================================
    // LOAD PHASE: load table using burst-friendly memcpy into both buffers
    // =========================================================
    memcpy((int*)l_table_0, (int*)table, sizeof(int) * n * n);
    memcpy((int*)l_table_1, (int*)table, sizeof(int) * n * n);

    // =========================================================
    // COMPUTE PHASE: Nussinov DP
    // We compute into l_table_0 (primary buffer).
    // =========================================================
    int i, j;

    for (i = n - 1; i >= 0; i--) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=180
        for (j = i + 1; j < n; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=179
#pragma HLS DEPENDENCE variable=l_table_0 inter false

            if (j - 1 >= 0)
                l_table_0[i][j] = ((l_table_0[i][j] >= l_table_0[i][j-1]) ? l_table_0[i][j] : l_table_0[i][j-1]);
            if (i + 1 < n)
                l_table_0[i][j] = ((l_table_0[i][j] >= l_table_0[i+1][j]) ? l_table_0[i][j] : l_table_0[i+1][j]);

            if (j - 1 >= 0 && i + 1 < n) {
                if (i < j - 1)
                    l_table_0[i][j] = ((l_table_0[i][j] >= l_table_0[i+1][j-1] + (((l_seq[i]) + (l_seq[j])) == 3 ? 1 : 0)) ? l_table_0[i][j] : l_table_0[i+1][j-1] + (((l_seq[i]) + (l_seq[j])) == 3 ? 1 : 0));
                else
                    l_table_0[i][j] = ((l_table_0[i][j] >= l_table_0[i+1][j-1]) ? l_table_0[i][j] : l_table_0[i+1][j-1]);
            }

            // Tiled inner k-loop
            int tij = l_table_0[i][j];

            compute_k_tile: for (int kt = i + 1; kt < j; kt += TILE_SIZE) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=12
                int kt_end = (kt + TILE_SIZE < j) ? (kt + TILE_SIZE) : j;

                int tile_row[TILE_SIZE];
                int tile_col[TILE_SIZE];
#pragma HLS ARRAY_PARTITION variable=tile_row complete dim=1
#pragma HLS ARRAY_PARTITION variable=tile_col complete dim=1

                load_tile_k: for (int kk = 0; kk < TILE_SIZE; kk++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL factor=8
#pragma HLS DEPENDENCE variable=l_table_0 inter false
                    int k_idx = kt + kk;
                    if (k_idx < kt_end) {
                        tile_row[kk] = l_table_0[i][k_idx];
                        tile_col[kk] = l_table_0[k_idx + 1][j];
                    } else {
                        tile_row[kk] = 0;
                        tile_col[kk] = 0;
                    }
                }

                compute_tile_k: for (int kk = 0; kk < TILE_SIZE; kk++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL factor=8
#pragma HLS DEPENDENCE variable=tile_row inter false
#pragma HLS DEPENDENCE variable=tile_col inter false
                    int k_idx = kt + kk;
                    if (k_idx < kt_end) {
                        int val = tile_row[kk] + tile_col[kk];
                        tij = (tij >= val) ? tij : val;
                    }
                }
            }
            l_table_0[i][j] = tij;
        }
    }

    // =========================================================
    // STORE PHASE with DOUBLE BUFFERING
    // =========================================================

    int num_row_tiles = (n + TILE_SIZE - 1) / TILE_SIZE;

    // Staging buffers for double buffering the store phase
    int store_buf_0[TILE_SIZE][N];
    int store_buf_1[TILE_SIZE][N];
#pragma HLS ARRAY_PARTITION variable=store_buf_0 cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=store_buf_0 cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=store_buf_1 cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=store_buf_1 cyclic factor=16 dim=2

    // Preload first tile (t=0) into store_buf_0
    {
        int ti = 0;
        int ti_end = (ti + TILE_SIZE < n) ? (ti + TILE_SIZE) : n;
        preload_store_i: for (int ii = ti; ii < ti_end; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=16
            preload_store_j: for (int jj = 0; jj < n; jj++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=180
#pragma HLS UNROLL factor=8
                store_buf_0[ii - ti][jj] = l_table_0[ii][jj];
            }
        }
    }

    // Double-buffered store loop using burst-friendly memcpy writes
    store_db_outer: for (int t = 0; t < num_row_tiles; t++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=12

        int ti_curr = t * TILE_SIZE;
        int ti_curr_end = (ti_curr + TILE_SIZE < n) ? (ti_curr + TILE_SIZE) : n;
        int rows_curr = ti_curr_end - ti_curr;

        int ti_next = (t + 1) * TILE_SIZE;
        int ti_next_end = (ti_next + TILE_SIZE < n) ? (ti_next + TILE_SIZE) : n;

        int buf_curr = t % 2;
        int buf_next = 1 - buf_curr;

        // STORE current tile (from buf_curr) to global memory via burst memcpy
        if (buf_curr == 0) {
            memcpy((int*)table + ti_curr * n, (int*)store_buf_0,
                   sizeof(int) * rows_curr * n);
        } else {
            memcpy((int*)table + ti_curr * n, (int*)store_buf_1,
                   sizeof(int) * rows_curr * n);
        }

        // Load next tile into buf_next (if there is a next tile)
        if (ti_next < n) {
            load_next_i: for (int ii = ti_next; ii < ti_next_end; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=16
                load_next_j: for (int jj = 0; jj < n; jj++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=180
#pragma HLS UNROLL factor=8
                    if (buf_next == 0)
                        store_buf_0[ii - ti_next][jj] = l_table_0[ii][jj];
                    else
                        store_buf_1[ii - ti_next][jj] = l_table_0[ii][jj];
                }
            }
        }
    }
}

} // extern "C"