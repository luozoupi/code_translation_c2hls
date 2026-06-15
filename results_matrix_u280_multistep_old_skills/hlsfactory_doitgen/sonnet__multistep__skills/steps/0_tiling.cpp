#include "doitgen.h"
#include <string.h>

// Tile sizes
#define TILE_R 5   // NR=25, so 5 tiles of 5
#define TILE_Q 4   // NQ=20, so 5 tiles of 4

extern "C" {

// Load a tile of A: A[r0..r0+TILE_R][q0..q0+TILE_Q][0..NP]
static void load_A_tile(
    double A[NR][NQ][NP],
    double l_A[TILE_R][TILE_Q][NP],
    int r0, int q0)
{
    for (int r = 0; r < TILE_R; r++)
        for (int q = 0; q < TILE_Q; q++)
            for (int p = 0; p < NP; p++) {
#pragma HLS PIPELINE II=1
                l_A[r][q][p] = A[r0 + r][q0 + q][p];
            }
}

// Load C4 into local buffer
static void load_C4_tile(
    double C4[NP][NP],
    double l_C4[NP][NP])
{
    for (int s = 0; s < NP; s++)
        for (int p = 0; p < NP; p++) {
#pragma HLS PIPELINE II=1
            l_C4[s][p] = C4[s][p];
        }
}

// Compute: for each (r,q) in tile, compute sum[p] = sum_s A[r][q][s]*C4[s][p]
//          then write back into l_A[r][q][p]
static void compute_tile(
    double l_A[TILE_R][TILE_Q][NP],
    double l_C4[NP][NP],
    double l_sum[NP])
{
    for (int r = 0; r < TILE_R; r++)
        for (int q = 0; q < TILE_Q; q++) {
            // Compute sum[p] = sum_s A[r][q][s] * C4[s][p]
            for (int p = 0; p < NP; p++) {
#pragma HLS PIPELINE II=1
                double acc = 0.0;
                for (int s = 0; s < NP; s++) {
#pragma HLS UNROLL factor=4
                    acc += l_A[r][q][s] * l_C4[s][p];
                }
                l_sum[p] = acc;
            }
            // Write sum back to tile buffer
            for (int p = 0; p < NP; p++) {
#pragma HLS PIPELINE II=1
                l_A[r][q][p] = l_sum[p];
            }
        }
}

// Store tile of A back to global memory
static void store_A_tile(
    double A[NR][NQ][NP],
    double l_A[TILE_R][TILE_Q][NP],
    int r0, int q0)
{
    for (int r = 0; r < TILE_R; r++)
        for (int q = 0; q < TILE_Q; q++)
            for (int p = 0; p < NP; p++) {
#pragma HLS PIPELINE II=1
                A[r0 + r][q0 + q][p] = l_A[r][q][p];
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

    // Local tile buffers (much smaller than full A)
    double l_A[TILE_R][TILE_Q][NP];
    double l_C4[NP][NP];
    double l_sum[NP];

#pragma HLS ARRAY_PARTITION variable=l_A   cyclic factor=4 dim=3
#pragma HLS ARRAY_PARTITION variable=l_C4  cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=l_C4  cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=l_sum complete dim=1

    // Load C4 once — reused across all tiles
    load_C4_tile(C4, l_C4);

    // Tile loop over r and q dimensions
    for (int r0 = 0; r0 < NR; r0 += TILE_R) {
        for (int q0 = 0; q0 < NQ; q0 += TILE_Q) {

            // --- LOAD phase: bring tile of A into local buffer ---
            load_A_tile(A, l_A, r0, q0);

            // --- COMPUTE phase: operate entirely on local buffers ---
            compute_tile(l_A, l_C4, l_sum);

            // --- STORE phase: write tile of A back to global memory ---
            store_A_tile(A, l_A, r0, q0);
        }
    }

    // Store final sum (last tile's l_sum) back to global memory
    store_sum:
    for (int p = 0; p < NP; p++) {
#pragma HLS PIPELINE II=1
        sum[p] = l_sum[p];
    }
}

} // extern "C"