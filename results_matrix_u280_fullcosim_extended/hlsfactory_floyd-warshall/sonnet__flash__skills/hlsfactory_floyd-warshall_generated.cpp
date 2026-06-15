#include "floyd-warshall.h"

extern "C" {

void kernel_floyd_warshall(int path[N + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=path offset=slave bundle=gmem depth=32400
#pragma HLS INTERFACE s_axilite port=path bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;

    // Local buffer for the k-th row (path[k][*]) to avoid repeated global reads
    int path_k[N];
#pragma HLS ARRAY_PARTITION variable=path_k complete dim=1

    int i, j, k;

    for (k = 0; k < n; k++)
    {
        // Load row k into local buffer
        load_k: for (j = 0; j < n; j++)
        {
#pragma HLS PIPELINE II=1
            path_k[j] = path[k][j];
        }

        // Update all paths using row k
        loop_i: for (i = 0; i < n; i++)
        {
            // Cache path[i][k] as a scalar to avoid repeated global memory reads in j-loop
            int path_ik = path[i][k];

            loop_j: for (j = 0; j < n; j++)
            {
#pragma HLS PIPELINE II=1
                int old_val  = path[i][j];
                int new_val  = path_ik + path_k[j];
                path[i][j]   = old_val < new_val ? old_val : new_val;
            }
        }
    }
}

} // extern "C"