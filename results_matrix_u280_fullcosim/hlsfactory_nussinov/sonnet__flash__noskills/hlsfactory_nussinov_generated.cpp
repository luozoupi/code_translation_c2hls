#include "nussinov.h"

extern "C" {

void kernel_nussinov( char seq[ N + 0],
                       int table[ N + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=seq   offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=table offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=seq    bundle=control
#pragma HLS INTERFACE s_axilite port=table  bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;

    // Local copies for fast on-chip access
    char  l_seq[N];
    int   l_table[N][N];

#pragma HLS ARRAY_PARTITION variable=l_seq   complete dim=1
#pragma HLS ARRAY_PARTITION variable=l_table cyclic  factor=8 dim=2

    // Load seq from global memory
    load_seq: for (int ii = 0; ii < n; ii++) {
#pragma HLS PIPELINE II=1
        l_seq[ii] = seq[ii];
    }

    // Load table from global memory
    load_table: for (int ii = 0; ii < n; ii++) {
        for (int jj = 0; jj < n; jj++) {
#pragma HLS PIPELINE II=1
            l_table[ii][jj] = table[ii][jj];
        }
    }

    // Main computation (algorithm unchanged)
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

            for (k = i+1; k < j; k++) {
#pragma HLS PIPELINE II=1
                l_table[i][j] = ((l_table[i][j] >= l_table[i][k] + l_table[k+1][j]) ? l_table[i][j] : l_table[i][k] + l_table[k+1][j]);
            }
        }
    }

    // Write back table to global memory
    store_table: for (int ii = 0; ii < n; ii++) {
        for (int jj = 0; jj < n; jj++) {
#pragma HLS PIPELINE II=1
            table[ii][jj] = l_table[ii][jj];
        }
    }
}

} // extern "C"