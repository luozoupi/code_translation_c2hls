#include "bicg.h"
#include <string.h>

// Tile sizes
#define TILE_I 16   // tile over rows (N dimension)
#define TILE_J 116  // full M dimension (process all columns per tile)

// Unroll factor for the inner compute loop
#define UNROLL_J 4

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

    // Double-buffered tile buffers for A rows and r values
    double A_tile_0[TILE_I][TILE_J];
    double A_tile_1[TILE_I][TILE_J];
    double r_tile_0[TILE_I];
    double r_tile_1[TILE_I];

    // Partition A_tile buffers dim=2 by factor=8 (covers unroll and pipeline)
#pragma HLS ARRAY_PARTITION variable=A_tile_0 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=A_tile_1 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=r_tile_0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=r_tile_1 complete dim=1

    // Full-size local buffers for p, s, q (reused across tiles)
    double p_local[M];
    double s_local[M];
    double q_local[N];

    // Partition p_local and s_local by 8 (superset of unroll factor 4, conflict-free)
#pragma HLS ARRAY_PARTITION variable=p_local cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=s_local cyclic factor=8 dim=1

    // -------------------------------------------------------
    // LOAD phase: bring p into local buffer (used by all tiles)
    // -------------------------------------------------------
    load_p: for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=116 max=116
        p_local[j] = p[j];
    }

    // Initialize s_local to zero (accumulates across all row tiles)
    init_s: for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=116 max=116
        s_local[j] = 0.0;
    }

    // -------------------------------------------------------
    // Pre-load the first tile (tile 0) into buffer 0 before the main loop
    // -------------------------------------------------------
    {
        int ii = 0;
        int tile_rows_0 = (ii + TILE_I <= N) ? TILE_I : (N - ii);

        preload_r: for (int ti = 0; ti < tile_rows_0; ti++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
            r_tile_0[ti] = r[ii + ti];
        }

        preload_A: for (int ti = 0; ti < tile_rows_0; ti++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
            preload_A_j: for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=116 max=116
                A_tile_0[ti][j] = A[ii + ti][j];
            }
        }
    }

    // -------------------------------------------------------
    // Tiled computation with double buffering:
    //   - Ping buffer (ping=0): A_tile_0, r_tile_0
    //   - Pong buffer (ping=1): A_tile_1, r_tile_1
    //   - While computing from buffer X, load next tile into buffer 1-X
    // -------------------------------------------------------
    int num_tiles = (N + TILE_I - 1) / TILE_I;

    tile_i: for (int tile = 0; tile < num_tiles; tile++) {
#pragma HLS LOOP_TRIPCOUNT min=8 max=8
        int ii      = tile * TILE_I;
        int ping    = tile & 1;  // 0 for even tiles, 1 for odd tiles

        // Determine actual tile height for the CURRENT tile
        int tile_rows_cur = (ii + TILE_I <= N) ? TILE_I : (N - ii);

        // -------------------------------------------------------
        // LOAD phase: load NEXT tile into the opposite buffer
        // (runs concurrently with compute below via task overlap)
        // -------------------------------------------------------
        int next_tile = tile + 1;
        int ii_next   = next_tile * TILE_I;

        if (next_tile < num_tiles) {
            int tile_rows_next = (ii_next + TILE_I <= N) ? TILE_I : (N - ii_next);

            if (ping == 0) {
                // Currently computing from buffer 0, load next into buffer 1
                load_r_next_0: for (int ti = 0; ti < tile_rows_next; ti++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
                    r_tile_1[ti] = r[ii_next + ti];
                }
                load_A_next_0: for (int ti = 0; ti < tile_rows_next; ti++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
                    load_A_next_0_j: for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=116 max=116
                        A_tile_1[ti][j] = A[ii_next + ti][j];
                    }
                }
            } else {
                // Currently computing from buffer 1, load next into buffer 0
                load_r_next_1: for (int ti = 0; ti < tile_rows_next; ti++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
                    r_tile_0[ti] = r[ii_next + ti];
                }
                load_A_next_1: for (int ti = 0; ti < tile_rows_next; ti++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
                    load_A_next_1_j: for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=116 max=116
                        A_tile_0[ti][j] = A[ii_next + ti][j];
                    }
                }
            }
        }

        // -------------------------------------------------------
        // COMPUTE phase: operate on current tile from the active buffer
        // -------------------------------------------------------
        if (ping == 0) {
            // Compute from buffer 0
            compute_tile_i_0: for (int ti = 0; ti < tile_rows_cur; ti++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
                double q_acc = 0.0;
                double r_i   = r_tile_0[ti];

                compute_tile_j_0: for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=116 max=116
#pragma HLS DEPENDENCE variable=s_local inter false
#pragma HLS DEPENDENCE variable=A_tile_0 inter false
#pragma HLS DEPENDENCE variable=p_local inter false
                    double a_ij = A_tile_0[ti][j];
                    s_local[j] += r_i * a_ij;
                    q_acc       += a_ij * p_local[j];
                }

                q_local[ii + ti] = q_acc;
            }
        } else {
            // Compute from buffer 1
            compute_tile_i_1: for (int ti = 0; ti < tile_rows_cur; ti++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
                double q_acc = 0.0;
                double r_i   = r_tile_1[ti];

                compute_tile_j_1: for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=116 max=116
#pragma HLS DEPENDENCE variable=s_local inter false
#pragma HLS DEPENDENCE variable=A_tile_1 inter false
#pragma HLS DEPENDENCE variable=p_local inter false
                    double a_ij = A_tile_1[ti][j];
                    s_local[j] += r_i * a_ij;
                    q_acc       += a_ij * p_local[j];
                }

                q_local[ii + ti] = q_acc;
            }
        }

        // --- Store phase: write q values for this tile to global memory ---
        store_q_tile: for (int ti = 0; ti < tile_rows_cur; ti++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
            q[ii + ti] = q_local[ii + ti];
        }
    }

    // -------------------------------------------------------
    // STORE phase: write fully accumulated s back to global memory
    // -------------------------------------------------------
    store_s: for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=116 max=116
        s[j] = s_local[j];
    }
}

} // extern "C"