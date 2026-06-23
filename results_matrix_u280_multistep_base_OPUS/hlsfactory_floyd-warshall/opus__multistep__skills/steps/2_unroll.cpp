#include "floyd-warshall.h"
#include <cstring>


void kernel_floyd_warshall(
			   int path[ N + 0][N + 0])
{
#pragma HLS INLINE off

    const int n = N;

    int i, j, k;

    // Local buffer for the k-th row (reused across all i)
    int row_k[N];
#pragma HLS ARRAY_PARTITION variable=row_k cyclic factor=4 dim=1
    // Local buffer for the current row i being processed
    int row_i[N];
#pragma HLS ARRAY_PARTITION variable=row_i cyclic factor=4 dim=1

    for (k = 0; k < n; k++)
    {
        // ---- LOAD phase: stage the k-th row into local memory ----
        load_row_k:
        for (j = 0; j < n; j++)
        {
#pragma HLS LOOP_TRIPCOUNT min=N max=N
#pragma HLS PIPELINE II=1
            row_k[j] = path[k][j];
        }

        for (i = 0; i < n; i++)
        {
            // ---- LOAD phase: stage row i into local memory ----
            load_row_i:
            for (j = 0; j < n; j++)
            {
#pragma HLS LOOP_TRIPCOUNT min=N max=N
#pragma HLS PIPELINE II=1
                row_i[j] = path[i][j];
            }

            // scalar reused across all j for this (k,i)
            int path_ik = row_i[k];

            // ---- COMPUTE phase: operate on local buffers ----
            compute_row:
            for (j = 0; j < n; j++)
            {
#pragma HLS LOOP_TRIPCOUNT min=N max=N
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS DEPENDENCE variable=row_i inter false
                int candidate = path_ik + row_k[j];
                row_i[j] = row_i[j] < candidate ? row_i[j] : candidate;
            }

            // ---- STORE phase: write row i back to global memory ----
            store_row_i:
            for (j = 0; j < n; j++)
            {
#pragma HLS LOOP_TRIPCOUNT min=N max=N
#pragma HLS PIPELINE II=1
                path[i][j] = row_i[j];
            }
        }
    }
}

extern "C" {
void workload(int path[N + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=path offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=path bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    kernel_floyd_warshall(path);
}
}