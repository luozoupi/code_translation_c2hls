#include "gramschmidt.h"
#include <cmath>

#define TILE_M 16

// Load a tile of A rows [row_start, row_start+TILE_M) into local buffer
static void load_A_tile(double A_global[M][N], double A_local[TILE_M][N], int row_start) {
    for (int i = 0; i < TILE_M; i++) {
#pragma HLS PIPELINE II=1
        for (int j = 0; j < N; j++) {
            int gi = row_start + i;
            A_local[i][j] = (gi < M) ? A_global[gi][j] : 0.0;
        }
    }
}

// Store a tile of A back to global memory
static void store_A_tile(double A_global[M][N], double A_local[TILE_M][N], int row_start) {
    for (int i = 0; i < TILE_M; i++) {
#pragma HLS PIPELINE II=1
        for (int j = 0; j < N; j++) {
            int gi = row_start + i;
            if (gi < M) A_global[gi][j] = A_local[i][j];
        }
    }
}

// Load a tile of Q rows [row_start, row_start+TILE_M) into local buffer
static void load_Q_tile(double Q_global[M][N], double Q_local[TILE_M][N], int row_start) {
    for (int i = 0; i < TILE_M; i++) {
#pragma HLS PIPELINE II=1
        for (int j = 0; j < N; j++) {
            int gi = row_start + i;
            Q_local[i][j] = (gi < M) ? Q_global[gi][j] : 0.0;
        }
    }
}

// Store a tile of Q back to global memory
static void store_Q_tile(double Q_global[M][N], double Q_local[TILE_M][N], int row_start) {
    for (int i = 0; i < TILE_M; i++) {
#pragma HLS PIPELINE II=1
        for (int j = 0; j < N; j++) {
            int gi = row_start + i;
            if (gi < M) Q_global[gi][j] = Q_local[i][j];
        }
    }
}

extern "C" {

void kernel_gramschmidt(
    double A[M + 0][N + 0],
    double R[N + 0][N + 0],
    double Q[M + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=R offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=Q offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=R bundle=control
#pragma HLS INTERFACE s_axilite port=Q bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Tiled local buffers: TILE_M x N strips instead of full M x N
    double A_tile[TILE_M][N];
    double Q_tile[TILE_M][N];
    double R_local[N][N];

    // Partition dim=2 (columns) cyclic factor=8 for parallel column access
    // Also partition dim=1 (rows) completely since TILE_M=16 is small, for row-parallel access
#pragma HLS ARRAY_PARTITION variable=A_tile cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=A_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=Q_tile cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=Q_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=R_local cyclic factor=8 dim=2

    const int num_tiles = (M + TILE_M - 1) / TILE_M;

    // Initialize R to zero
    init_R_outer: for (int i = 0; i < N; i++) {
#pragma HLS LOOP_TRIPCOUNT min=80 max=80
        init_R_inner: for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=80 max=80
            R_local[i][j] = 0.0;
        }
    }

    for (int k = 0; k < N; k++) {
#pragma HLS LOOP_TRIPCOUNT min=80 max=80

        // ------ Step (a): Compute norm of A[:,k] tiled ------
        double nrm = 0.0;
        norm_tile_loop: for (int t = 0; t < num_tiles; t++) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=4
            int row_start = t * TILE_M;

            // Load tile of A
            load_A_tile(A, A_tile, row_start);

            // Accumulate norm over this tile
            norm_inner: for (int i = 0; i < TILE_M; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS DEPENDENCE variable=A_tile inter false
                int gi = row_start + i;
                if (gi < M) {
                    nrm += A_tile[i][k] * A_tile[i][k];
                }
            }
        }
        R_local[k][k] = sqrt(nrm);
        double rkk = R_local[k][k];

        // ------ Step (b): Compute Q[:,k] and store tiled ------
        q_tile_loop: for (int t = 0; t < num_tiles; t++) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=4
            int row_start = t * TILE_M;

            // Load tile of A (re-load for this pass)
            load_A_tile(A, A_tile, row_start);

            // Compute Q tile column k
            q_inner: for (int i = 0; i < TILE_M; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS DEPENDENCE variable=A_tile inter false
#pragma HLS DEPENDENCE variable=Q_tile inter false
                int gi = row_start + i;
                if (gi < M) {
                    Q_tile[i][k] = A_tile[i][k] / rkk;
                }
            }

            // Store Q tile back to global
            store_Q_tile(Q, Q_tile, row_start);
        }

        // ------ Steps (c) and (d): Update R and A for j > k ------
        update_j_loop: for (int j = k + 1; j < N; j++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=79
            double rkj = 0.0;

            // (c) Accumulate R[k][j] tiled
            r_tile_loop: for (int t = 0; t < num_tiles; t++) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=4
                int row_start = t * TILE_M;

                // Load Q tile column k (need all columns for indexing)
                load_Q_tile(Q, Q_tile, row_start);

                // Load A tile
                load_A_tile(A, A_tile, row_start);

                // Accumulate R[k][j]
                r_inner: for (int i = 0; i < TILE_M; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS DEPENDENCE variable=A_tile inter false
#pragma HLS DEPENDENCE variable=Q_tile inter false
                    int gi = row_start + i;
                    if (gi < M) {
                        rkj += Q_tile[i][k] * A_tile[i][j];
                    }
                }
            }
            R_local[k][j] = rkj;

            // (d) Update A[:,j] tiled
            a_tile_loop: for (int t = 0; t < num_tiles; t++) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=4
                int row_start = t * TILE_M;

                // Load Q tile
                load_Q_tile(Q, Q_tile, row_start);

                // Load A tile
                load_A_tile(A, A_tile, row_start);

                // Update A tile column j
                a_inner: for (int i = 0; i < TILE_M; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS DEPENDENCE variable=A_tile inter false
#pragma HLS DEPENDENCE variable=Q_tile inter false
                    int gi = row_start + i;
                    if (gi < M) {
                        A_tile[i][j] = A_tile[i][j] - Q_tile[i][k] * rkj;
                    }
                }

                // Store updated A tile back
                store_A_tile(A, A_tile, row_start);
            }
        }
    }

    // ------------------------------------------------------------------
    // PHASE 2: Store R back to global memory
    // ------------------------------------------------------------------
    store_R_outer: for (int i = 0; i < N; i++) {
#pragma HLS LOOP_TRIPCOUNT min=80 max=80
        store_R_inner: for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=80 max=80
#pragma HLS DEPENDENCE variable=R_local inter false
            R[i][j] = R_local[i][j];
        }
    }
}

} // extern "C"