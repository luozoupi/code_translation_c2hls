#include "floyd-warshall.h"

extern "C" {

void kernel_floyd_warshall(
    int path[N + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=path offset=slave bundle=gmem depth=32400
#pragma HLS INTERFACE s_axilite port=path bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local on-chip copy to avoid repeated off-chip access and enable pipelining
    int local_path[N][N];
#pragma HLS ARRAY_PARTITION variable=local_path cyclic factor=8 dim=2

    // Load from global memory
    load_i: for (int i = 0; i < N; i++) {
        load_j: for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            local_path[i][j] = path[i][j];
        }
    }

    // Floyd-Warshall on local array
    fw_k: for (int k = 0; k < N; k++) {
        fw_i: for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
            fw_j: for (int j = 0; j < N; j++) {
                int via_k = local_path[i][k] + local_path[k][j];
                local_path[i][j] = local_path[i][j] < via_k ?
                    local_path[i][j] : via_k;
            }
        }
    }

    // Write back to global memory
    store_i: for (int i = 0; i < N; i++) {
        store_j: for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            path[i][j] = local_path[i][j];
        }
    }

}

} // extern "C"