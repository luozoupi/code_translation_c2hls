#include "ludcmp.h"
#include <string.h>

#define TILE_SIZE 16

// Load a tile of rows [row_start, row_start+TILE_SIZE) of A into local tile buffer
static void load_tile_rows(double A[N][N], double tile[TILE_SIZE][N], int row_start) {
    for (int i = 0; i < TILE_SIZE && (row_start + i) < N; i++) {
        for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            tile[i][j] = A[row_start + i][j];
        }
    }
}

static void store_tile_rows(double A[N][N], double tile[TILE_SIZE][N], int row_start) {
    for (int i = 0; i < TILE_SIZE && (row_start + i) < N; i++) {
        for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            A[row_start + i][j] = tile[i][j];
        }
    }
}

void kernel_ludcmp(
           double A[N + 0][N + 0],
           double b[N + 0],
           double x[N + 0],
           double y[N + 0])
{
    #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi port=y offset=slave bundle=gmem3
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

    #pragma HLS ARRAY_PARTITION variable=A_local cyclic factor=16 dim=2
    #pragma HLS ARRAY_PARTITION variable=y_local complete dim=1
    #pragma HLS ARRAY_PARTITION variable=x_local complete dim=1
    #pragma HLS ARRAY_PARTITION variable=b_local complete dim=1

    // ----------------------------------------------------------------
    // LOAD PHASE: Load A in tiles of TILE_SIZE rows
    // ----------------------------------------------------------------
    load_tiles: for (int tile_row = 0; tile_row < N; tile_row += TILE_SIZE) {
        #pragma HLS LOOP_TRIPCOUNT min=8 max=8
        // Local tile buffer for a chunk of rows
        double A_tile[TILE_SIZE][N];
        #pragma HLS ARRAY_PARTITION variable=A_tile cyclic factor=16 dim=2

        // Load tile from global memory
        load_tile_inner_i: for (int i = 0; i < TILE_SIZE && (tile_row + i) < N; i++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=16
            load_tile_inner_j: for (int j = 0; j < N; j++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=120 max=120
                A_tile[i][j] = A[tile_row + i][j];
            }
        }

        // Copy tile into full local buffer
        copy_tile_to_local_i: for (int i = 0; i < TILE_SIZE && (tile_row + i) < N; i++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=16
            copy_tile_to_local_j: for (int j = 0; j < N; j++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=120 max=120
                A_local[tile_row + i][j] = A_tile[i][j];
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

                // Buffer this tile's columns from row i and the diagonal tile
                double row_tile[TILE_SIZE];
                double diag_tile[TILE_SIZE];
                #pragma HLS ARRAY_PARTITION variable=row_tile complete dim=1
                #pragma HLS ARRAY_PARTITION variable=diag_tile complete dim=1

                // Load tile of A[i][tile_j..tile_j_end]
                load_row_tile: for (int jj = 0; jj < TILE_SIZE && (tile_j + jj) < tile_j_end; jj++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=16
                    row_tile[jj] = A_local[i][tile_j + jj];
                    diag_tile[jj] = A_local[tile_j + jj][tile_j + jj];
                }

                // Compute lower triangular updates for this j tile
                lu_lower_j: for (j = tile_j; j < tile_j_end; j++) {
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=16
                    w = A_local[i][j];
                    lu_lower_k: for (k = 0; k < j; k++) {
                        #pragma HLS PIPELINE II=1
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
                #pragma HLS LOOP_TRIPCOUNT min=0 max=120
                #pragma HLS DEPENDENCE variable=A_local inter false
                #pragma HLS DEPENDENCE variable=x_local inter false
                w -= A_local[i][j] * x_local[j];
            }
            x_local[i] = w / A_local[i][i];
        }
    }

    // ----------------------------------------------------------------
    // STORE PHASE: Write results back in tiles of TILE_SIZE rows
    // ----------------------------------------------------------------
    store_tiles: for (int tile_row = 0; tile_row < N; tile_row += TILE_SIZE) {
        #pragma HLS LOOP_TRIPCOUNT min=8 max=8
        // Local tile buffer for a chunk of rows
        double A_out_tile[TILE_SIZE][N];
        #pragma HLS ARRAY_PARTITION variable=A_out_tile cyclic factor=16 dim=2

        // Copy from full local buffer into output tile
        copy_local_to_tile_i: for (int i = 0; i < TILE_SIZE && (tile_row + i) < N; i++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=16
            copy_local_to_tile_j: for (int j = 0; j < N; j++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=120 max=120
                A_out_tile[i][j] = A_local[tile_row + i][j];
            }
        }

        // Store tile to global memory
        store_tile_inner_i: for (int i = 0; i < TILE_SIZE && (tile_row + i) < N; i++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=16
            store_tile_inner_j: for (int j = 0; j < N; j++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=120 max=120
                A[tile_row + i][j] = A_out_tile[i][j];
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