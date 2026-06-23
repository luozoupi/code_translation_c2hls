#include "trisolv.h"
#include <cstring>

#define TILE 256

void kernel_trisolv(
		    double L[ N + 0][N + 0],
		    double x[ N + 0],
		    double b[ N + 0])
{
    const int n = N;

    int i, j, jt;

    // Local buffers for reusable working set
    double x_local[N];
    double b_local[N];
#pragma HLS BIND_STORAGE variable=x_local type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=b_local type=ram_2p impl=bram

    // ---- Load phase: stage b into local memory ----
    load_b: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        b_local[i] = b[i];
    }

    // ---- Compute phase: operate on local buffers ----
    loop_i: for (i = 0; i < n; i++)
    {
        double acc = b_local[i];

        // Local tile buffer for one row segment of L
        double L_tile[TILE];
#pragma HLS BIND_STORAGE variable=L_tile type=ram_2p impl=bram

        // Process the reduction in tiles of TILE elements
        tile_j: for (jt = 0; jt < i; jt += TILE) {
            int jmax = jt + TILE;
            if (jmax > i) jmax = i;
            int len = jmax - jt;

            // Load a tile of row L[i][jt..jmax)
            load_L: for (j = 0; j < len; j++) {
#pragma HLS PIPELINE II=1
                L_tile[j] = L[i][jt + j];
            }

            // Compute reduction over local tile
            compute_j: for (j = 0; j < len; j++) {
#pragma HLS PIPELINE II=1
                acc -= L_tile[j] * x_local[jt + j];
            }
        }

        x_local[i] = acc / L[i][i];
    }

    // ---- Store phase: write results back to global memory ----
    store_x: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        x[i] = x_local[i];
    }
}


extern "C" {
void workload(
		    double L[ N + 0][N + 0],
		    double x[ N + 0],
		    double b[ N + 0])
{
#pragma HLS INTERFACE m_axi port=L offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=L bundle=control
#pragma HLS INTERFACE s_axilite port=x bundle=control
#pragma HLS INTERFACE s_axilite port=b bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  kernel_trisolv(L, x, b);
}
}