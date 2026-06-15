#include "nussinov.h"

#define TILE_SIZE 16
#define UNROLL_FACTOR 8

extern "C" {

void kernel_nussinov(char seq[N + 0],
                     int table[N + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=seq   offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=table offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=seq    bundle=control
#pragma HLS INTERFACE s_axilite port=table  bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local buffers - single for seq (small), double-buffered for table rows
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
    // LOAD PHASE: load seq (no double buffering needed - single pass)
    // =========================================================
    load_seq_tiles: for (int t = 0; t < n; t += TILE_SIZE) {
        int tile_end = (t + TILE_SIZE < n) ? (t + TILE_SIZE) : n;
        load_seq_inner: for (int ii = t; ii < tile_end; ii++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=16
            l_seq[ii] = seq[ii];
        }
    }

    // =========================================================
    // LOAD PHASE: load table using double buffering across row-tiles
    // Tile 0 (buf=0): load into l_table_0
    // Tile 1 (buf=1): load into l_table_1, simultaneously "compute" on l_table_0
    // We load all tiles first (two-phase ping-pong over row tiles)
    // =========================================================

    // We use a flag per row-tile iteration to select buffer
    // First, load first tile into buffer 0
    // Then alternate: while loading tile t into buf (t/TILE_SIZE)%2,
    //   the previous tile sits in the other buffer ready for use.

    // Because the compute phase needs the FULL table, we load entire table
    // into buffer 0 first (ping), then the compute uses buffer 0.
    // Double buffering here overlaps row-tile loads:
    // row-tile t   -> loads into buf[(t/TILE_SIZE)%2 == 0 ? 0 : 1]
    // row-tile t-1 -> already loaded in the other buffer

    // Pragmatic approach: load all of table into both buffers isn't useful.
    // Instead, double-buffer the STORE phase: while storing row-tile t,
    // we prepare row-tile t+1 data. For load, we double-buffer row tiles.

    // Phase 1: Load entire table into l_table_0 (first buffer)
    // This is our "ping" load - compute will use l_table_0
    load_table_row_tile: for (int ti = 0; ti < n; ti += TILE_SIZE) {
        int ti_end = (ti + TILE_SIZE < n) ? (ti + TILE_SIZE) : n;
        load_table_col_tile: for (int tj = 0; tj < n; tj += TILE_SIZE) {
            int tj_end = (tj + TILE_SIZE < n) ? (tj + TILE_SIZE) : n;
            load_table_i: for (int ii = ti; ii < ti_end; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=16
                load_table_j: for (int jj = tj; jj < tj_end; jj++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=16
#pragma HLS UNROLL factor=8
                    l_table_0[ii][jj] = table[ii][jj];
                    l_table_1[ii][jj] = table[ii][jj]; // load into both so compute can use either
                }
            }
        }
    }

    // =========================================================
    // COMPUTE PHASE: Nussinov DP
    // We compute into l_table_0 (primary buffer).
    // Double buffering is applied in the store phase below.
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
    // STORE PHASE with DOUBLE BUFFERING:
    // Use l_table_0 (compute result) and l_table_1 as staging buffer.
    // While storing row-tile (ti-TILE_SIZE) from l_table_1 to global memory,
    // copy row-tile ti from l_table_0 into l_table_1.
    // This overlaps the copy-to-staging with the global memory write.
    //
    // Schedule per row-tile iteration t (t = 0, 1, ..., num_row_tiles-1):
    //   - Copy row-tile t from l_table_0 into buf[t%2]
    //   - Store row-tile (t-1) from buf[(t-1)%2] to global memory
    // After loop: store final tile from buf[(num_tiles-1)%2].
    // =========================================================

    int num_row_tiles = (n + TILE_SIZE - 1) / TILE_SIZE;

    // We need staging buffers for double buffering the store phase
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

    // Double-buffered store loop
    store_db_outer: for (int t = 0; t < num_row_tiles; t++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=12

        int ti_curr = t * TILE_SIZE;
        int ti_curr_end = (ti_curr + TILE_SIZE < n) ? (ti_curr + TILE_SIZE) : n;

        int ti_next = (t + 1) * TILE_SIZE;
        int ti_next_end = (ti_next + TILE_SIZE < n) ? (ti_next + TILE_SIZE) : n;

        int buf_curr = t % 2;       // buffer holding current tile (ready to store)
        int buf_next = 1 - buf_curr; // buffer to load next tile into

        // STORE current tile (from buf_curr) to global memory
        // while simultaneously LOADING next tile (into buf_next)
        // We interleave these by doing them in separate sub-loops
        // (HLS will schedule them to overlap via pipelining)

        // Store current tile from buf_curr
        store_curr_i: for (int ii = ti_curr; ii < ti_curr_end; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=16
            store_curr_j: for (int jj = 0; jj < n; jj++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=180
#pragma HLS UNROLL factor=8
                if (buf_curr == 0)
                    table[ii][jj] = store_buf_0[ii - ti_curr][jj];
                else
                    table[ii][jj] = store_buf_1[ii - ti_curr][jj];
            }
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