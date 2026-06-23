#include "nussinov.h"
#include <cstring>

#define TILE 256

extern "C" {

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
            // Stage the two operand vectors of the reduction into local tile
            // buffers, then perform the max-reduction on the local tiles.
            const int k_start = i + 1;
            const int k_end   = j;          // exclusive

            // Local tile buffers for the working set of this reduction.
            int tileA[TILE];  // l_table[i][k]
            int tileB[TILE];  // l_table[k+1][j]
#pragma HLS ARRAY_PARTITION variable=tileA cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=tileB cyclic factor=8

        tile_loop:
            for (int kt = k_start; kt < k_end; kt += TILE) {
                int chunk = k_end - kt;
                if (chunk > TILE) chunk = TILE;

                // ---- load phase: stage tile into local buffers ----
            load_tile:
                for (int p = 0; p < chunk; p++) {
#pragma HLS PIPELINE II=1
                    int kk = kt + p;
                    tileA[p] = l_table[i][kk];
                    tileB[p] = l_table[kk + 1][j];
                }

                // ---- compute phase: operate on local tile buffers ----
            compute_tile:
                for (int p = 0; p < chunk; p++) {
#pragma HLS PIPELINE II=1
                    int cand = tileA[p] + tileB[p];
                    acc = (acc >= cand) ? acc : cand;
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