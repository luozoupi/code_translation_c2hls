#include "trisolv.h"
#include <cstring>

#define TILE 256

// Load a tile of row L[i][jt..jt+len) into the selected buffer
static void load_L_tile(double L[N + 0][N + 0], double L_tile_1[TILE],
                        double L_tile_2[TILE], int i, int jt, int len, int flag)
{
    if (flag == 0) {
        load_L0: for (int j = 0; j < len; j++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
            L_tile_1[j] = L[i][jt + j];
        }
    } else {
        load_L1: for (int j = 0; j < len; j++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
            L_tile_2[j] = L[i][jt + j];
        }
    }
}

// Compute reduction over the selected tile buffer
static void compute_L_tile(double L_tile_1[TILE], double L_tile_2[TILE],
                           double x_local[N], double &acc, int jt, int len,
                           int flag)
{
    if (flag == 0) {
        compute_j0: for (int j = 0; j < len; j++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS DEPENDENCE variable=x_local inter false
#pragma HLS DEPENDENCE variable=acc inter false
            acc -= L_tile_1[j] * x_local[jt + j];
        }
    } else {
        compute_j1: for (int j = 0; j < len; j++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS DEPENDENCE variable=x_local inter false
#pragma HLS DEPENDENCE variable=acc inter false
            acc -= L_tile_2[j] * x_local[jt + j];
        }
    }
}

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
#pragma HLS ARRAY_PARTITION variable=x_local cyclic factor=4 dim=1

    // ---- Load phase: stage b into local memory ----
    load_b: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        b_local[i] = b[i];
    }

    // ---- Compute phase: operate on local buffers ----
    loop_i: for (i = 0; i < n; i++)
    {
#pragma HLS LOOP_TRIPCOUNT min=N max=N
        double acc = b_local[i];

        // Double-buffered tile storage for row segments of L
        double L_tile_1[TILE];
        double L_tile_2[TILE];
#pragma HLS BIND_STORAGE variable=L_tile_1 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=L_tile_2 type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=L_tile_1 cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=L_tile_2 cyclic factor=4 dim=1

        // Number of tiles for this row
        int ntiles = (i + TILE - 1) / TILE;

        // Pre-load the first tile (prologue)
        if (ntiles > 0) {
            int len0 = TILE;
            if (len0 > i) len0 = i;
            load_L_tile(L, L_tile_1, L_tile_2, i, 0, len0, 0);
        }

        // Software-pipelined loop: load tile k+1 while computing tile k
        tile_j: for (int t = 0; t < ntiles; t++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=1
            int flag = t % 2;
            jt = t * TILE;
            int jmax = jt + TILE;
            if (jmax > i) jmax = i;
            int len = jmax - jt;

            // Prefetch next tile into the other buffer
            int next_flag = (t + 1) % 2;
            int next_jt = (t + 1) * TILE;
            int next_jmax = next_jt + TILE;
            if (next_jmax > i) next_jmax = i;
            int next_len = next_jmax - next_jt;

            if (t + 1 < ntiles) {
                load_L_tile(L, L_tile_1, L_tile_2, i, next_jt, next_len, next_flag);
            }

            // Compute on the current tile
            compute_L_tile(L_tile_1, L_tile_2, x_local, acc, jt, len, flag);
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