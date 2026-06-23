#include "nussinov.h"
#include <cstring>

#define TILE 256

extern "C" {

static void load_tile_db(int l_table[N][N], int tileA[TILE], int tileB[TILE],
                         int i, int j, int kt, int chunk)
{
load_tile:
    for (int p = 0; p < chunk; p++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
        int kk = kt + p;
        tileA[p] = l_table[i][kk];
        tileB[p] = l_table[kk + 1][j];
    }
}

static int compute_tile_db(int tileA[TILE], int tileB[TILE], int acc, int chunk)
{
compute_tile:
    for (int p = 0; p < chunk; p++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
        int cand = tileA[p] + tileB[p];
        acc = (acc >= cand) ? acc : cand;
    }
    return acc;
}

void kernel_nussinov( char seq[ N + 0],
			   int table[ N + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=seq   offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=table offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=seq    bundle=control
#pragma HLS INTERFACE s_axilite port=table  bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;

    // Stage data into local buffers to enable reuse and partitioned parallel access.
    static char  l_seq[N];
    static int   l_table[N][N];
#pragma HLS ARRAY_PARTITION variable=l_table cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_table cyclic factor=8 dim=1

    // Load inputs into local memory.
load_seq:
    for (int t = 0; t < n; t++) {
#pragma HLS PIPELINE II=1
        l_seq[t] = seq[t];
    }
load_table:
    for (int r = 0; r < n; r++) {
        for (int c = 0; c < n; c++) {
#pragma HLS PIPELINE II=1
            l_table[r][c] = table[r][c];
        }
    }

    int i, j, k;

    for (i = n-1; i >= 0; i--) {
        for (j = i+1; j < n; j++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=180

            if (j-1 >= 0)
                l_table[i][j] = ((l_table[i][j] >= l_table[i][j-1]) ? l_table[i][j] : l_table[i][j-1]);
            if (i+1 < n)
                l_table[i][j] = ((l_table[i][j] >= l_table[i+1][j]) ? l_table[i][j] : l_table[i+1][j]);

            if (j-1 >= 0 && i+1 < n) {
                if (i < j-1)
                    l_table[i][j] = ((l_table[i][j] >= l_table[i+1][j-1]+(((l_seq[i])+(l_seq[j])) == 3 ? 1 : 0)) ? l_table[i][j] : l_table[i+1][j-1]+(((l_seq[i])+(l_seq[j])) == 3 ? 1 : 0));
                else
                    l_table[i][j] = ((l_table[i][j] >= l_table[i+1][j-1]) ? l_table[i][j] : l_table[i+1][j-1]);
            }

            int acc = l_table[i][j];

            // ---- Tiled reduction over k in [i+1, j) ----
            const int k_start = i + 1;
            const int k_end   = j;          // exclusive

            // Double-buffered local tile buffers (ping-pong).
            int tileA_1[TILE];  int tileB_1[TILE];
            int tileA_2[TILE];  int tileB_2[TILE];
#pragma HLS ARRAY_PARTITION variable=tileA_1 cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=tileB_1 cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=tileA_2 cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=tileB_2 cyclic factor=8

            // Number of tiles for this reduction range.
            int total = k_end - k_start;
            int num_tiles = (total + TILE - 1) / TILE;
            if (num_tiles < 0) num_tiles = 0;

            // Software-pipelined ping-pong over tiles:
            // load tile (t) into one buffer set while computing tile (t-1)
            // from the other buffer set.
        tile_loop:
            for (int t = 0; t <= num_tiles; t++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=2

                int flag = t & 1;  // selects which buffer to LOAD into this iter

                // ---- LOAD phase for tile t (if it exists) ----
                if (t < num_tiles) {
                    int kt = k_start + t * TILE;
                    int chunk = k_end - kt;
                    if (chunk > TILE) chunk = TILE;
                    if (flag == 0)
                        load_tile_db(l_table, tileA_1, tileB_1, i, j, kt, chunk);
                    else
                        load_tile_db(l_table, tileA_2, tileB_2, i, j, kt, chunk);
                }

                // ---- COMPUTE phase for tile t-1 (already loaded last iter) ----
                if (t > 0) {
                    int pt = t - 1;
                    int kt_c = k_start + pt * TILE;
                    int chunk_c = k_end - kt_c;
                    if (chunk_c > TILE) chunk_c = TILE;
                    int pflag = pt & 1;  // buffer that tile t-1 was loaded into
                    if (pflag == 0)
                        acc = compute_tile_db(tileA_1, tileB_1, acc, chunk_c);
                    else
                        acc = compute_tile_db(tileA_2, tileB_2, acc, chunk_c);
                }
            }

            l_table[i][j] = acc;
        }
    }

    // Write results back to global memory.
store_table:
    for (int r = 0; r < n; r++) {
        for (int c = 0; c < n; c++) {
#pragma HLS PIPELINE II=1
            table[r][c] = l_table[r][c];
        }
    }
}

} // extern "C"