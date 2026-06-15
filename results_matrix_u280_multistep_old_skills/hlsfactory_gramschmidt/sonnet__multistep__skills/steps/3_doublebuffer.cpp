#include "gramschmidt.h"
#include <cmath>

#define TILE_M 16

// Load a tile of A rows [row_start, row_start+TILE_M) into local buffer (double-buffered)
static void load_A_tile(double A_global[M][N],
                         double A_local0[TILE_M][N],
                         double A_local1[TILE_M][N],
                         int row_start, int buf_sel) {
    if (buf_sel == 0) {
        for (int i = 0; i < TILE_M; i++) {
#pragma HLS PIPELINE II=1
            for (int j = 0; j < N; j++) {
#pragma HLS UNROLL factor=8
                int gi = row_start + i;
                A_local0[i][j] = (gi < M) ? A_global[gi][j] : 0.0;
            }
        }
    } else {
        for (int i = 0; i < TILE_M; i++) {
#pragma HLS PIPELINE II=1
            for (int j = 0; j < N; j++) {
#pragma HLS UNROLL factor=8
                int gi = row_start + i;
                A_local1[i][j] = (gi < M) ? A_global[gi][j] : 0.0;
            }
        }
    }
}

// Store a tile of A back to global memory (double-buffered)
static void store_A_tile(double A_global[M][N],
                          double A_local0[TILE_M][N],
                          double A_local1[TILE_M][N],
                          int row_start, int buf_sel) {
    if (buf_sel == 0) {
        for (int i = 0; i < TILE_M; i++) {
#pragma HLS PIPELINE II=1
            for (int j = 0; j < N; j++) {
#pragma HLS UNROLL factor=8
                int gi = row_start + i;
                if (gi < M) A_global[gi][j] = A_local0[i][j];
            }
        }
    } else {
        for (int i = 0; i < TILE_M; i++) {
#pragma HLS PIPELINE II=1
            for (int j = 0; j < N; j++) {
#pragma HLS UNROLL factor=8
                int gi = row_start + i;
                if (gi < M) A_global[gi][j] = A_local1[i][j];
            }
        }
    }
}

// Load a tile of Q rows [row_start, row_start+TILE_M) into local buffer (double-buffered)
static void load_Q_tile(double Q_global[M][N],
                         double Q_local0[TILE_M][N],
                         double Q_local1[TILE_M][N],
                         int row_start, int buf_sel) {
    if (buf_sel == 0) {
        for (int i = 0; i < TILE_M; i++) {
#pragma HLS PIPELINE II=1
            for (int j = 0; j < N; j++) {
#pragma HLS UNROLL factor=8
                int gi = row_start + i;
                Q_local0[i][j] = (gi < M) ? Q_global[gi][j] : 0.0;
            }
        }
    } else {
        for (int i = 0; i < TILE_M; i++) {
#pragma HLS PIPELINE II=1
            for (int j = 0; j < N; j++) {
#pragma HLS UNROLL factor=8
                int gi = row_start + i;
                Q_local1[i][j] = (gi < M) ? Q_global[gi][j] : 0.0;
            }
        }
    }
}

// Store a tile of Q back to global memory (double-buffered)
static void store_Q_tile(double Q_global[M][N],
                          double Q_local0[TILE_M][N],
                          double Q_local1[TILE_M][N],
                          int row_start, int buf_sel) {
    if (buf_sel == 0) {
        for (int i = 0; i < TILE_M; i++) {
#pragma HLS PIPELINE II=1
            for (int j = 0; j < N; j++) {
#pragma HLS UNROLL factor=8
                int gi = row_start + i;
                if (gi < M) Q_global[gi][j] = Q_local0[i][j];
            }
        }
    } else {
        for (int i = 0; i < TILE_M; i++) {
#pragma HLS PIPELINE II=1
            for (int j = 0; j < N; j++) {
#pragma HLS UNROLL factor=8
                int gi = row_start + i;
                if (gi < M) Q_global[gi][j] = Q_local1[i][j];
            }
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

    // Double-buffered local arrays: _0 and _1 ping-pong buffers
    double A_tile_0[TILE_M][N];
    double A_tile_1[TILE_M][N];
    double Q_tile_0[TILE_M][N];
    double Q_tile_1[TILE_M][N];
    double R_local[N][N];

    // Partition dim=2 (columns) cyclic factor=8 for parallel column access
    // Also partition dim=1 (rows) completely since TILE_M=16 is small
#pragma HLS ARRAY_PARTITION variable=A_tile_0 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=A_tile_0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=A_tile_1 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=A_tile_1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=Q_tile_0 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=Q_tile_0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=Q_tile_1 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=Q_tile_1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=R_local cyclic factor=8 dim=2

    const int num_tiles = (M + TILE_M - 1) / TILE_M;

    // Initialize R to zero
    init_R_outer: for (int i = 0; i < N; i++) {
#pragma HLS LOOP_TRIPCOUNT min=80 max=80
        init_R_inner: for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=80 max=80
#pragma HLS UNROLL factor=8
            R_local[i][j] = 0.0;
        }
    }

    for (int k = 0; k < N; k++) {
#pragma HLS LOOP_TRIPCOUNT min=80 max=80

        // ------ Step (a): Compute norm of A[:,k] tiled with double buffering ------
        double nrm = 0.0;

        // Pre-load first tile into buffer 0
        load_A_tile(A, A_tile_0, A_tile_1, 0, 0);

        norm_tile_loop: for (int t = 0; t < num_tiles; t++) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=4
            int row_start = t * TILE_M;
            int cur_buf = t & 1;       // current compute buffer
            int nxt_buf = 1 - cur_buf; // next load buffer

            // Pre-load next tile while we compute current tile
            if (t + 1 < num_tiles) {
                int next_row_start = (t + 1) * TILE_M;
                load_A_tile(A, A_tile_0, A_tile_1, next_row_start, nxt_buf);
            }

            // Accumulate norm over current tile
            norm_inner: for (int i = 0; i < TILE_M; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL factor=16
                int gi = row_start + i;
                if (gi < M) {
                    double val;
                    if (cur_buf == 0) {
                        val = A_tile_0[i][k];
                    } else {
                        val = A_tile_1[i][k];
                    }
                    nrm += val * val;
                }
            }
        }
        R_local[k][k] = sqrt(nrm);
        double rkk = R_local[k][k];

        // ------ Step (b): Compute Q[:,k] and store tiled with double buffering ------

        // Pre-load first tile of A into buffer 0 for Q computation
        load_A_tile(A, A_tile_0, A_tile_1, 0, 0);

        q_tile_loop: for (int t = 0; t < num_tiles; t++) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=4
            int row_start = t * TILE_M;
            int cur_buf = t & 1;
            int nxt_buf = 1 - cur_buf;

            // Pre-load next A tile while computing current
            if (t + 1 < num_tiles) {
                int next_row_start = (t + 1) * TILE_M;
                load_A_tile(A, A_tile_0, A_tile_1, next_row_start, nxt_buf);
            }

            // Compute Q tile column k from current A tile
            q_inner: for (int i = 0; i < TILE_M; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL factor=16
#pragma HLS DEPENDENCE variable=Q_tile_0 inter false
#pragma HLS DEPENDENCE variable=Q_tile_1 inter false
                int gi = row_start + i;
                if (gi < M) {
                    double a_val = (cur_buf == 0) ? A_tile_0[i][k] : A_tile_1[i][k];
                    if (cur_buf == 0) {
                        Q_tile_0[i][k] = a_val / rkk;
                    } else {
                        Q_tile_1[i][k] = a_val / rkk;
                    }
                }
            }

            // Store Q tile back to global using current buffer
            store_Q_tile(Q, Q_tile_0, Q_tile_1, row_start, cur_buf);
        }

        // ------ Steps (c) and (d): Update R and A for j > k ------
        update_j_loop: for (int j = k + 1; j < N; j++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=79
            double rkj = 0.0;

            // (c) Accumulate R[k][j] tiled with double buffering
            // Pre-load first tiles
            load_Q_tile(Q, Q_tile_0, Q_tile_1, 0, 0);
            load_A_tile(A, A_tile_0, A_tile_1, 0, 0);

            r_tile_loop: for (int t = 0; t < num_tiles; t++) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=4
                int row_start = t * TILE_M;
                int cur_buf = t & 1;
                int nxt_buf = 1 - cur_buf;

                // Pre-load next tiles
                if (t + 1 < num_tiles) {
                    int next_row_start = (t + 1) * TILE_M;
                    load_Q_tile(Q, Q_tile_0, Q_tile_1, next_row_start, nxt_buf);
                    load_A_tile(A, A_tile_0, A_tile_1, next_row_start, nxt_buf);
                }

                // Accumulate R[k][j] from current buffers
                r_inner: for (int i = 0; i < TILE_M; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL factor=16
                    int gi = row_start + i;
                    if (gi < M) {
                        double q_val = (cur_buf == 0) ? Q_tile_0[i][k] : Q_tile_1[i][k];
                        double a_val = (cur_buf == 0) ? A_tile_0[i][j] : A_tile_1[i][j];
                        rkj += q_val * a_val;
                    }
                }
            }
            R_local[k][j] = rkj;

            // (d) Update A[:,j] tiled with double buffering
            // Pre-load first tiles
            load_Q_tile(Q, Q_tile_0, Q_tile_1, 0, 0);
            load_A_tile(A, A_tile_0, A_tile_1, 0, 0);

            a_tile_loop: for (int t = 0; t < num_tiles; t++) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=4
                int row_start = t * TILE_M;
                int cur_buf = t & 1;
                int nxt_buf = 1 - cur_buf;

                // Pre-load next tiles
                if (t + 1 < num_tiles) {
                    int next_row_start = (t + 1) * TILE_M;
                    load_Q_tile(Q, Q_tile_0, Q_tile_1, next_row_start, nxt_buf);
                    load_A_tile(A, A_tile_0, A_tile_1, next_row_start, nxt_buf);
                }

                // Update A tile column j from current buffers
                a_inner: for (int i = 0; i < TILE_M; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL factor=16
#pragma HLS DEPENDENCE variable=A_tile_0 inter false
#pragma HLS DEPENDENCE variable=A_tile_1 inter false
#pragma HLS DEPENDENCE variable=Q_tile_0 inter false
#pragma HLS DEPENDENCE variable=Q_tile_1 inter false
                    int gi = row_start + i;
                    if (gi < M) {
                        if (cur_buf == 0) {
                            double q_val = Q_tile_0[i][k];
                            A_tile_0[i][j] = A_tile_0[i][j] - q_val * rkj;
                        } else {
                            double q_val = Q_tile_1[i][k];
                            A_tile_1[i][j] = A_tile_1[i][j] - q_val * rkj;
                        }
                    }
                }

                // Store updated A tile back using current buffer
                store_A_tile(A, A_tile_0, A_tile_1, row_start, cur_buf);
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
#pragma HLS UNROLL factor=8
#pragma HLS DEPENDENCE variable=R_local inter false
            R[i][j] = R_local[i][j];
        }
    }
}

} // extern "C"