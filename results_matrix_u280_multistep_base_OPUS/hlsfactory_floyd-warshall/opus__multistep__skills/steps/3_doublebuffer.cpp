#include "floyd-warshall.h"
#include <cstring>


// Load row i into the selected buffer
static void load_row(int path[N + 0][N + 0], int row_i_1[N], int row_i_2[N],
                     int i, int n, int flag)
{
#pragma HLS INLINE off
    load_row_i:
    for (int j = 0; j < n; j++)
    {
#pragma HLS LOOP_TRIPCOUNT min=N max=N
#pragma HLS PIPELINE II=1
        if (flag == 0)
            row_i_1[j] = path[i][j];
        else
            row_i_2[j] = path[i][j];
    }
}

// Compute on the selected buffer and store result back to global memory
static void compute_store(int path[N + 0][N + 0], int row_k[N],
                          int row_i_1[N], int row_i_2[N],
                          int i, int k, int n, int flag)
{
#pragma HLS INLINE off
    int path_ik = (flag == 0) ? row_i_1[k] : row_i_2[k];

    compute_row:
    for (int j = 0; j < n; j++)
    {
#pragma HLS LOOP_TRIPCOUNT min=N max=N
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS DEPENDENCE variable=row_i_1 inter false
#pragma HLS DEPENDENCE variable=row_i_2 inter false
        if (flag == 0)
        {
            int candidate = path_ik + row_k[j];
            int v = row_i_1[j];
            v = v < candidate ? v : candidate;
            row_i_1[j] = v;
            path[i][j] = v;
        }
        else
        {
            int candidate = path_ik + row_k[j];
            int v = row_i_2[j];
            v = v < candidate ? v : candidate;
            row_i_2[j] = v;
            path[i][j] = v;
        }
    }
}


void kernel_floyd_warshall(
			   int path[ N + 0][N + 0])
{
#pragma HLS INLINE off

    const int n = N;

    int i, j, k;

    // Local buffer for the k-th row (reused across all i)
    int row_k[N];
#pragma HLS ARRAY_PARTITION variable=row_k cyclic factor=4 dim=1

    // Two local buffers for the row i being processed (ping-pong)
    int row_i_1[N];
#pragma HLS ARRAY_PARTITION variable=row_i_1 cyclic factor=4 dim=1
    int row_i_2[N];
#pragma HLS ARRAY_PARTITION variable=row_i_2 cyclic factor=4 dim=1

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

        // ---- Double-buffered pipeline over rows i ----
        // Prologue: load row 0 into buffer 1 (flag 0)
        load_row(path, row_i_1, row_i_2, 0, n, 0);

        row_loop:
        for (i = 0; i < n; i++)
        {
#pragma HLS LOOP_TRIPCOUNT min=N max=N
            int flag = i % 2;

            // Load next row (i+1) into the opposite buffer while we
            // compute/store the current row i. These two operations
            // touch different ping-pong buffers, allowing overlap.
            if (i + 1 < n)
                load_row(path, row_i_1, row_i_2, i + 1, n, (i + 1) % 2);

            compute_store(path, row_k, row_i_1, row_i_2, i, k, n, flag);
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