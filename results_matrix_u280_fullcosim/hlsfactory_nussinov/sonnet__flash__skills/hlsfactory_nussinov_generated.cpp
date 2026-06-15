#include "nussinov.h"

extern "C" {

void kernel_nussinov( char seq[ N + 0],
                      int table[ N + 0][N + 0])
{
    #pragma HLS INTERFACE m_axi port=seq    offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=table  offset=slave bundle=gmem1
    #pragma HLS INTERFACE s_axilite port=seq    bundle=control
    #pragma HLS INTERFACE s_axilite port=table  bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;

    // Local copies to avoid repeated global memory accesses
    char l_seq[N];
    #pragma HLS ARRAY_PARTITION variable=l_seq complete dim=1

    int l_table[N][N];
    #pragma HLS ARRAY_PARTITION variable=l_table cyclic factor=4 dim=1
    #pragma HLS ARRAY_PARTITION variable=l_table cyclic factor=4 dim=2

    // Load seq into local buffer
    load_seq: for (int ii = 0; ii < n; ii++) {
        #pragma HLS PIPELINE II=1
        l_seq[ii] = seq[ii];
    }

    // Load table into local buffer
    load_table_i: for (int ii = 0; ii < n; ii++) {
        load_table_j: for (int jj = 0; jj < n; jj++) {
            #pragma HLS PIPELINE II=1
            l_table[ii][jj] = table[ii][jj];
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
                    l_table[i][j] = ((l_table[i][j] >= l_table[i+1][j-1] + (((l_seq[i]) + (l_seq[j])) == 3 ? 1 : 0)) ? l_table[i][j] : l_table[i+1][j-1] + (((l_seq[i]) + (l_seq[j])) == 3 ? 1 : 0));
                else
                    l_table[i][j] = ((l_table[i][j] >= l_table[i+1][j-1]) ? l_table[i][j] : l_table[i+1][j-1]);
            }

            int t_ij = l_table[i][j];
            for (k = i+1; k < j; k++) {
                #pragma HLS PIPELINE II=1
                int candidate = l_table[i][k] + l_table[k+1][j];
                t_ij = ((t_ij >= candidate) ? t_ij : candidate);
            }
            l_table[i][j] = t_ij;
        }
    }

    // Store table back to global memory
    store_table_i: for (int ii = 0; ii < n; ii++) {
        store_table_j: for (int jj = 0; jj < n; jj++) {
            #pragma HLS PIPELINE II=1
            table[ii][jj] = l_table[ii][jj];
        }
    }
}

} // extern "C"