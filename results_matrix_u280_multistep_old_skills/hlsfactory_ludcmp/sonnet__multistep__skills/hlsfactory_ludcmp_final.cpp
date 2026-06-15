#include "ludcmp.h"
#include <string.h>

#define TILE_SIZE 16
#define UNROLL_FACTOR 8

extern "C" {
void kernel_ludcmp(
           double A[N + 0][N + 0],
           double b[N + 0],
           double x[N + 0],
           double y[N + 0])
{
    #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 max_read_burst_length=256 max_write_burst_length=256
    #pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem1 max_read_burst_length=256 max_write_burst_length=256
    #pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem2 max_read_burst_length=256 max_write_burst_length=256
    #pragma HLS INTERFACE m_axi port=y offset=slave bundle=gmem3 max_read_burst_length=256 max_write_burst_length=256
    #pragma HLS INTERFACE s_axilite port=A bundle=control
    #pragma HLS INTERFACE s_axilite port=b bundle=control
    #pragma HLS INTERFACE s_axilite port=x bundle=control
    #pragma HLS INTERFACE s_axilite port=y bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // Full on-chip buffer for A (needed for LU decomposition cross-row dependencies)
    double A_local[N][N];
    double b_local[N];
    double x_local[N];
    double y_local[N];

    #pragma HLS ARRAY_PARTITION variable=A_local cyclic factor=8 dim=2
    #pragma HLS ARRAY_PARTITION variable=y_local cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=x_local cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=b_local cyclic factor=8 dim=1

    // Double buffers for tile loading (ping-pong)
    double A_tile_0[TILE_SIZE][N];
    double A_tile_1[TILE_SIZE][N];
    #pragma HLS ARRAY_PARTITION variable=A_tile_0 cyclic factor=8 dim=2
    #pragma HLS ARRAY_PARTITION variable=A_tile_1 cyclic factor=8 dim=2

    // Double buffers for tile storing (ping-pong)
    double A_out_tile_0[TILE_SIZE][N];
    double A_out_tile_1[TILE_SIZE][N];
    #pragma HLS ARRAY_PARTITION variable=A_out_tile_0 cyclic factor=8 dim=2
    #pragma HLS ARRAY_PARTITION variable=A_out_tile_1 cyclic factor=8 dim=2

    const int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    // ----------------------------------------------------------------
    // LOAD PHASE with double buffering:
    // Load tile k+1 from global memory while copying tile k into A_local
    // ----------------------------------------------------------------

    // Pre-load the first tile into buffer 0
    {
        int tile_row = 0;
        for (int i = 0; i < TILE_SIZE && (tile_row + i) < N; i++) {
            for (int j = 0; j < N; j++) {
                #pragma HLS PIPELINE II=1
                A_tile_0[i][j] = A[tile_row + i][j];
            }
        }
    }

    load_tiles: for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        #pragma HLS LOOP_TRIPCOUNT min=8 max=8
        int tile_row = tile_idx * TILE_SIZE;
        int buf_sel = tile_idx % 2; // which buffer currently holds data to copy

        // Pre-fetch next tile into the other buffer while copying current tile
        int next_tile_row = tile_row + TILE_SIZE;

        if (tile_idx < num_tiles - 1) {
            // First copy current tile to A_local
            copy_tile_to_local_i: for (int i = 0; i < TILE_SIZE && (tile_row + i) < N; i++) {
                #pragma HLS LOOP_TRIPCOUNT min=1 max=16
                copy_tile_to_local_j: for (int j = 0; j < N; j++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=120 max=120
                    if (buf_sel == 0) {
                        A_local[tile_row + i][j] = A_tile_0[i][j];
                    } else {
                        A_local[tile_row + i][j] = A_tile_1[i][j];
                    }
                }
            }
            // Load next tile into opposite buffer
            int next_buf = 1 - buf_sel;
            load_next_tile_i: for (int i = 0; i < TILE_SIZE && (next_tile_row + i) < N; i++) {
                #pragma HLS LOOP_TRIPCOUNT min=1 max=16
                load_next_tile_j: for (int j = 0; j < N; j++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=120 max=120
                    if (next_buf == 0) {
                        A_tile_0[i][j] = A[next_tile_row + i][j];
                    } else {
                        A_tile_1[i][j] = A[next_tile_row + i][j];
                    }
                }
            }
        } else {
            // Last tile: just copy to A_local
            copy_last_tile_i: for (int i = 0; i < TILE_SIZE && (tile_row + i) < N; i++) {
                #pragma HLS LOOP_TRIPCOUNT min=1 max=16
                copy_last_tile_j: for (int j = 0; j < N; j++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=120 max=120
                    if (buf_sel == 0) {
                        A_local[tile_row + i][j] = A_tile_0[i][j];
                    } else {
                        A_local[tile_row + i][j] = A_tile_1[i][j];
                    }
                }
            }
        }
    }

    // Load b, x, y in a single pipelined loop
    load_bxy: for (int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=120 max=120
        b_local[i] = b[i];
        x_local[i] = x[i];
        y_local[i] = y[i];
    }

    // ----------------------------------------------------------------
    // COMPUTE PHASE: LU decomposition, forward/back substitution
    // Operates entirely on local buffers A_local, b_local, x_local, y_local
    // ----------------------------------------------------------------
    const int n = N;
    int i, j, k;
    double w;

    // LU factorization - tiled by outer rows
    lu_outer_tile: for (int tile_i = 0; tile_i < n; tile_i += TILE_SIZE) {
        #pragma HLS LOOP_TRIPCOUNT min=8 max=8
        int tile_i_end = tile_i + TILE_SIZE;
        if (tile_i_end > n) tile_i_end = n;

        // Process each row in this tile
        lu_row: for (i = tile_i; i < tile_i_end; i++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=16

            // Lower triangular part: compute A[i][j] for j < i
            lu_lower_tile_j: for (int tile_j = 0; tile_j < i; tile_j += TILE_SIZE) {
                #pragma HLS LOOP_TRIPCOUNT min=0 max=8
                int tile_j_end = tile_j + TILE_SIZE;
                if (tile_j_end > i) tile_j_end = i;

                double row_tile[TILE_SIZE];
                double diag_tile[TILE_SIZE];
                #pragma HLS ARRAY_PARTITION variable=row_tile complete dim=1
                #pragma HLS ARRAY_PARTITION variable=diag_tile complete dim=1

                load_row_tile: for (int jj = 0; jj < TILE_SIZE && (tile_j + jj) < tile_j_end; jj++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=16
                    row_tile[jj] = A_local[i][tile_j + jj];
                    diag_tile[jj] = A_local[tile_j + jj][tile_j + jj];
                }

                lu_lower_j: for (j = tile_j; j < tile_j_end; j++) {
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=16
                    w = A_local[i][j];
                    lu_lower_k: for (k = 0; k < j; k++) {
                        #pragma HLS PIPELINE II=1
                        #pragma HLS UNROLL factor=UNROLL_FACTOR
                        #pragma HLS LOOP_TRIPCOUNT min=0 max=120
                        #pragma HLS DEPENDENCE variable=A_local inter false
                        w -= A_local[i][k] * A_local[k][j];
                    }
                    A_local[i][j] = w / A_local[j][j];
                }
            }

            // Upper triangular part: compute A[i][j] for j >= i
            lu_upper_tile_j: for (int tile_j = i; tile_j < n; tile_j += TILE_SIZE) {
                #pragma HLS LOOP_TRIPCOUNT min=1 max=8
                int tile_j_end = tile_j + TILE_SIZE;
                if (tile_j_end > n) tile_j_end = n;

                lu_upper_j: for (j = tile_j; j < tile_j_end; j++) {
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=16
                    w = A_local[i][j];
                    lu_upper_k: for (k = 0; k < i; k++) {
                        #pragma HLS PIPELINE II=1
                        #pragma HLS UNROLL factor=UNROLL_FACTOR
                        #pragma HLS LOOP_TRIPCOUNT min=0 max=120
                        #pragma HLS DEPENDENCE variable=A_local inter false
                        w -= A_local[i][k] * A_local[k][j];
                    }
                    A_local[i][j] = w;
                }
            }
        }
    }

    // Forward substitution: Ly = b
    fwd_tile: for (int tile_i = 0; tile_i < n; tile_i += TILE_SIZE) {
        #pragma HLS LOOP_TRIPCOUNT min=8 max=8
        int tile_i_end = tile_i + TILE_SIZE;
        if (tile_i_end > n) tile_i_end = n;

        fwd_i: for (i = tile_i; i < tile_i_end; i++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=16
            w = b_local[i];
            fwd_j: for (j = 0; j < i; j++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS UNROLL factor=UNROLL_FACTOR
                #pragma HLS LOOP_TRIPCOUNT min=0 max=120
                #pragma HLS DEPENDENCE variable=A_local inter false
                #pragma HLS DEPENDENCE variable=y_local inter false
                w -= A_local[i][j] * y_local[j];
            }
            y_local[i] = w;
        }
    }

    // Back substitution: Ux = y
    back_tile: for (int tile_i = n - 1; tile_i >= 0; tile_i -= TILE_SIZE) {
        #pragma HLS LOOP_TRIPCOUNT min=8 max=8
        int tile_i_start = tile_i - TILE_SIZE + 1;
        if (tile_i_start < 0) tile_i_start = 0;

        back_i: for (i = tile_i; i >= tile_i_start; i--) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=16
            w = y_local[i];
            back_j: for (j = i + 1; j < n; j++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS UNROLL factor=UNROLL_FACTOR
                #pragma HLS LOOP_TRIPCOUNT min=0 max=120
                #pragma HLS DEPENDENCE variable=A_local inter false
                #pragma HLS DEPENDENCE variable=x_local inter false
                w -= A_local[i][j] * x_local[j];
            }
            x_local[i] = w / A_local[i][i];
        }
    }

    // ----------------------------------------------------------------
    // STORE PHASE with double buffering:
    // Copy tile k from A_local into buffer while writing tile k-1 to global memory
    // ----------------------------------------------------------------

    // Pre-load first output tile into buffer 0
    {
        int tile_row = 0;
        for (int i = 0; i < TILE_SIZE && (tile_row + i) < N; i++) {
            for (int j = 0; j < N; j++) {
                #pragma HLS PIPELINE II=1
                A_out_tile_0[i][j] = A_local[tile_row + i][j];
            }
        }
    }

    store_tiles: for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        #pragma HLS LOOP_TRIPCOUNT min=8 max=8
        int tile_row = tile_idx * TILE_SIZE;
        int buf_sel = tile_idx % 2; // which buffer currently holds data to store

        int next_tile_row = tile_row + TILE_SIZE;

        if (tile_idx < num_tiles - 1) {
            int next_buf = 1 - buf_sel;

            // Load next tile into opposite buffer
            copy_next_out_tile_i: for (int i = 0; i < TILE_SIZE && (next_tile_row + i) < N; i++) {
                #pragma HLS LOOP_TRIPCOUNT min=1 max=16
                copy_next_out_tile_j: for (int j = 0; j < N; j++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=120 max=120
                    if (next_buf == 0) {
                        A_out_tile_0[i][j] = A_local[next_tile_row + i][j];
                    } else {
                        A_out_tile_1[i][j] = A_local[next_tile_row + i][j];
                    }
                }
            }

            // Store current tile to global memory
            store_tile_inner_i: for (int i = 0; i < TILE_SIZE && (tile_row + i) < N; i++) {
                #pragma HLS LOOP_TRIPCOUNT min=1 max=16
                store_tile_inner_j: for (int j = 0; j < N; j++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=120 max=120
                    if (buf_sel == 0) {
                        A[tile_row + i][j] = A_out_tile_0[i][j];
                    } else {
                        A[tile_row + i][j] = A_out_tile_1[i][j];
                    }
                }
            }
        } else {
            // Last tile: just store to global memory
            store_last_tile_i: for (int i = 0; i < TILE_SIZE && (tile_row + i) < N; i++) {
                #pragma HLS LOOP_TRIPCOUNT min=1 max=16
                store_last_tile_j: for (int j = 0; j < N; j++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=120 max=120
                    if (buf_sel == 0) {
                        A[tile_row + i][j] = A_out_tile_0[i][j];
                    } else {
                        A[tile_row + i][j] = A_out_tile_1[i][j];
                    }
                }
            }
        }
    }

    // Write back x and y
    store_xy: for (int ii = 0; ii < N; ii++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=120 max=120
        x[ii] = x_local[ii];
        y[ii] = y_local[ii];
    }
}
} // extern "C"