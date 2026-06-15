#include "bicg.h"
#include <string.h>

// Tile sizes
#define TILE_I 16   // tile over rows (N dimension)
#define TILE_J 116  // full M dimension (process all columns per tile)

extern "C" {

void kernel_bicg(
    double A[N + 0][M + 0],
    double s[M + 0],
    double q[N + 0],
    double p[M + 0],
    double r[N + 0])
{
#pragma HLS INTERFACE m_axi port=A      offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=s      offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=q      offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=p      offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=r      offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=A      bundle=control
#pragma HLS INTERFACE s_axilite port=s      bundle=control
#pragma HLS INTERFACE s_axilite port=q      bundle=control
#pragma HLS INTERFACE s_axilite port=p      bundle=control
#pragma HLS INTERFACE s_axilite port=r      bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Tile-sized local buffers for A rows and r values
    double A_tile[TILE_I][TILE_J];
    double r_tile[TILE_I];

#pragma HLS ARRAY_PARTITION variable=A_tile cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=r_tile complete dim=1

    // Full-size local buffers for p, s, q (reused across tiles)
    double p_local[M];
    double s_local[M];
    double q_local[N];

#pragma HLS ARRAY_PARTITION variable=p_local cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=s_local cyclic factor=8 dim=1

    // -------------------------------------------------------
    // LOAD phase: bring p into local buffer (used by all tiles)
    // -------------------------------------------------------
    load_p: for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
        p_local[j] = p[j];
    }

    // Initialize s_local to zero (accumulates across all row tiles)
    init_s: for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
        s_local[j] = 0.0;
    }

    // -------------------------------------------------------
    // Tiled computation: iterate over row tiles
    // -------------------------------------------------------
    tile_i: for (int ii = 0; ii < N; ii += TILE_I) {
        // Determine actual tile height (handle boundary)
        int tile_rows = (ii + TILE_I <= N) ? TILE_I : (N - ii);

        // --- Load tile: bring TILE_I rows of A and r into local buffers ---
        load_r_tile: for (int ti = 0; ti < tile_rows; ti++) {
#pragma HLS PIPELINE II=1
            r_tile[ti] = r[ii + ti];
        }

        load_A_tile: for (int ti = 0; ti < tile_rows; ti++) {
            load_A_tile_j: for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
                A_tile[ti][j] = A[ii + ti][j];
            }
        }

        // --- Compute phase: operate entirely on local buffers ---
        // For each row in the tile, accumulate into s_local and compute q
        compute_tile_i: for (int ti = 0; ti < tile_rows; ti++) {
            double q_acc = 0.0;
            double r_i   = r_tile[ti];

            compute_tile_j: for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
                double a_ij = A_tile[ti][j];
                s_local[j] += r_i * a_ij;
                q_acc       += a_ij * p_local[j];
            }

            // Store q for this row into local q buffer
            q_local[ii + ti] = q_acc;
        }

        // --- Store phase: write q values for this tile to global memory ---
        store_q_tile: for (int ti = 0; ti < tile_rows; ti++) {
#pragma HLS PIPELINE II=1
            q[ii + ti] = q_local[ii + ti];
        }
    }

    // -------------------------------------------------------
    // STORE phase: write fully accumulated s back to global memory
    // -------------------------------------------------------
    store_s: for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
        s[j] = s_local[j];
    }
}

} // extern "C"