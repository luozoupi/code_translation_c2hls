#include "floyd-warshall.h"

extern "C" {

void kernel_floyd_warshall(int path[N + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=path offset=slave bundle=gmem depth=32400
#pragma HLS INTERFACE s_axilite port=path bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;

    // Local buffers for the k-th row and k-th column to reduce global memory reads
    int row_k[N];
    int col_k[N];
#pragma HLS ARRAY_PARTITION variable=row_k complete dim=1
#pragma HLS ARRAY_PARTITION variable=col_k complete dim=1

    int i, j, k;

    for (k = 0; k < n; k++)
    {
        // Cache path[k][*] (k-th row) and path[*][k] (k-th column)
        load_row: for (j = 0; j < n; j++)
        {
#pragma HLS PIPELINE II=1
            row_k[j] = path[k][j];
        }

        load_col: for (i = 0; i < n; i++)
        {
#pragma HLS PIPELINE II=1
            col_k[i] = path[i][k];
        }

        // Update path[i][j] using cached row and column
        for (i = 0; i < n; i++)
        {
#pragma HLS PIPELINE II=1
            for (j = 0; j < n; j++)
            {
#pragma HLS UNROLL factor=4
                int candidate = col_k[i] + row_k[j];
                int cur = path[i][j];
                path[i][j] = cur < candidate ? cur : candidate;
            }
        }
    }
}

} // extern "C"