#include "doitgen.h"
#include <string.h>

// Tile sizes
#define TILE_R 5   // NR=25, so 5 tiles of 5
#define TILE_Q 4   // NQ=20, so 5 tiles of 4

// Total number of tiles
#define NUM_TILES_R (NR / TILE_R)
#define NUM_TILES_Q (NQ / TILE_Q)
#define NUM_TILES   (NUM_TILES_R * NUM_TILES_Q)

extern "C" {

// Load a tile of A: A[r0..r0+TILE_R][q0..q0+TILE_Q][0..NP]
// flag=0 -> load into l_A_0, flag=1 -> load into l_A_1
static void load_A_tile(
    double A[NR][NQ][NP],
    double l_A_0[TILE_R][TILE_Q][NP],
    double l_A_1[TILE_R][TILE_Q][NP],
    int r0, int q0, int flag)
{
#pragma HLS ARRAY_PARTITION variable=l_A_0 cyclic factor=8 dim=3
#pragma HLS ARRAY_PARTITION variable=l_A_1 cyclic factor=8 dim=3

    for (int r = 0; r < TILE_R; r++)
        for (int q = 0; q < TILE_Q; q++)
            for (int p = 0; p < NP; p++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=30 max=30
                double val = A[r0 + r][q0 + q][p];
                if (flag == 0)
                    l_A_0[r][q][p] = val;
                else
                    l_A_1[r][q][p] = val;
            }
}

// Load C4 into local buffer
static void load_C4_tile(
    double C4[NP][NP],
    double l_C4[NP][NP])
{
#pragma HLS ARRAY_PARTITION variable=l_C4 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_C4 cyclic factor=8 dim=2

    for (int s = 0; s < NP; s++)
        for (int p = 0; p < NP; p++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=30 max=30
            l_C4[s][p] = C4[s][p];
        }
}

// Compute: for each (r,q) in tile, compute sum[p] = sum_s A[r][q][s]*C4[s][p]
//          then write back into l_A[r][q][p]
// flag=0 -> compute from l_A_0, flag=1 -> compute from l_A_1
static void compute_tile(
    double l_A_0[TILE_R][TILE_Q][NP],
    double l_A_1[TILE_R][TILE_Q][NP],
    double l_C4[NP][NP],
    double l_sum[NP],
    int flag)
{
#pragma HLS ARRAY_PARTITION variable=l_A_0 cyclic factor=8 dim=3
#pragma HLS ARRAY_PARTITION variable=l_A_1 cyclic factor=8 dim=3
#pragma HLS ARRAY_PARTITION variable=l_C4  cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_C4  cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_sum complete dim=1

    for (int r = 0; r < TILE_R; r++) {
#pragma HLS LOOP_TRIPCOUNT min=5 max=5
        for (int q = 0; q < TILE_Q; q++) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=4
            // Compute sum[p] = sum_s A[r][q][s] * C4[s][p]
            for (int p = 0; p < NP; p++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=30 max=30
#pragma HLS UNROLL factor=2
#pragma HLS DEPENDENCE variable=l_sum inter false
                double acc = 0.0;
                for (int s = 0; s < NP; s++) {
#pragma HLS UNROLL factor=8
                    if (flag == 0)
                        acc += l_A_0[r][q][s] * l_C4[s][p];
                    else
                        acc += l_A_1[r][q][s] * l_C4[s][p];
                }
                l_sum[p] = acc;
            }
            // Write sum back to tile buffer
            for (int p = 0; p < NP; p++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=30 max=30
#pragma HLS UNROLL factor=2
#pragma HLS DEPENDENCE variable=l_A_0 inter false
#pragma HLS DEPENDENCE variable=l_A_1 inter false
                if (flag == 0)
                    l_A_0[r][q][p] = l_sum[p];
                else
                    l_A_1[r][q][p] = l_sum[p];
            }
        }
    }
}

// Store tile of A back to global memory
// flag=0 -> store from l_A_0, flag=1 -> store from l_A_1
static void store_A_tile(
    double A[NR][NQ][NP],
    double l_A_0[TILE_R][TILE_Q][NP],
    double l_A_1[TILE_R][TILE_Q][NP],
    int r0, int q0, int flag)
{
#pragma HLS ARRAY_PARTITION variable=l_A_0 cyclic factor=8 dim=3
#pragma HLS ARRAY_PARTITION variable=l_A_1 cyclic factor=8 dim=3

    for (int r = 0; r < TILE_R; r++)
        for (int q = 0; q < TILE_Q; q++)
            for (int p = 0; p < NP; p++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=30 max=30
                if (flag == 0)
                    A[r0 + r][q0 + q][p] = l_A_0[r][q][p];
                else
                    A[r0 + r][q0 + q][p] = l_A_1[r][q][p];
            }
}

void kernel_doitgen(  
		    double A[ NR + 0][NQ + 0][NP + 0],
		    double C4[ NP + 0][NP + 0],
		    double sum[ NP + 0])
{
#pragma HLS INTERFACE m_axi port=A   offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=C4  offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=sum offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=A      bundle=control
#pragma HLS INTERFACE s_axilite port=C4     bundle=control
#pragma HLS INTERFACE s_axilite port=sum    bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Double-buffered local tile buffers for A
    double l_A_0[TILE_R][TILE_Q][NP];
    double l_A_1[TILE_R][TILE_Q][NP];
    double l_C4[NP][NP];
    double l_sum[NP];

#pragma HLS ARRAY_PARTITION variable=l_A_0  cyclic factor=8 dim=3
#pragma HLS ARRAY_PARTITION variable=l_A_1  cyclic factor=8 dim=3
#pragma HLS ARRAY_PARTITION variable=l_C4   cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_C4   cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_sum  complete dim=1

    // Load C4 once — reused across all tiles
    load_C4_tile(C4, l_C4);

    // Flatten tile indices for double-buffer management
    // Total tiles = NUM_TILES_R * NUM_TILES_Q
    // We use a linear tile index to manage ping-pong easily

    // --- Pre-load first tile into buffer 0 ---
    int r0_first = 0;
    int q0_first = 0;
    load_A_tile(A, l_A_0, l_A_1, r0_first, q0_first, 0);

    // Tile loop over r and q dimensions (flattened)
    int tile_idx = 0;
    for (int r0 = 0; r0 < NR; r0 += TILE_R) {
#pragma HLS LOOP_TRIPCOUNT min=5 max=5
        for (int q0 = 0; q0 < NQ; q0 += TILE_Q) {
#pragma HLS LOOP_TRIPCOUNT min=5 max=5

            // Current buffer flag: tile_idx % 2
            int cur_flag = tile_idx % 2;

            // Determine next tile coordinates
            int next_q0 = q0 + TILE_Q;
            int next_r0 = r0;
            if (next_q0 >= NQ) {
                next_q0 = 0;
                next_r0 = r0 + TILE_R;
            }
            bool has_next = (next_r0 < NR);
            int next_flag = 1 - cur_flag;

            // --- COMPUTE phase: operate on current buffer ---
            compute_tile(l_A_0, l_A_1, l_C4, l_sum, cur_flag);

            // --- LOAD phase: pre-fetch next tile into the OTHER buffer ---
            if (has_next) {
                load_A_tile(A, l_A_0, l_A_1, next_r0, next_q0, next_flag);
            }

            // --- STORE phase: write current tile back to global memory ---
            store_A_tile(A, l_A_0, l_A_1, r0, q0, cur_flag);

            tile_idx++;
        }
    }

    // Store final sum (last tile's l_sum) back to global memory
    store_sum:
    for (int p = 0; p < NP; p++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=30 max=30
        sum[p] = l_sum[p];
    }
}

} // extern "C"