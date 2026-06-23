#include "floyd-warshall.h"

extern "C" {

void kernel_floyd_warshall(int path[N + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=path offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=path bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local copy for all computation to avoid repeated global memory accesses
    int local_path[N][N];
#pragma HLS ARRAY_PARTITION variable=local_path cyclic factor=8 dim=2

    // Load path into local buffer
    load_i: for (int i = 0; i < N; i++) {
        load_j: for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            local_path[i][j] = path[i][j];
        }
    }

    const int n = N;
    int i, j, k;

    // Floyd-Warshall: k carries true loop-carried dependency across all (i,j)
    for (k = 0; k < n; k++) {
        // Cache the k-th row: path[k][j] accessed for all i
        int row_k[N];
#pragma HLS ARRAY_PARTITION variable=row_k cyclic factor=8 dim=1
        cache_k: for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            row_k[j] = local_path[k][j];
        }

        for (i = 0; i < n; i++) {
            // Cache path[i][k] as scalar for the inner j loop
            int path_ik = local_path[i][k];
            compute_j: for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
                int through_k = path_ik + row_k[j];
                int cur = local_path[i][j];
                local_path[i][j] = (cur < through_k) ? cur : through_k;
            }
        }
    }

    // Store result back
    store_i: for (int i = 0; i < N; i++) {
        store_j: for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            path[i][j] = local_path[i][j];
        }
    }
}

} // extern "C"