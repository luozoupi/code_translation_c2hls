#include "lu.h"
#include <string.h>

// Tile size for blocked LU decomposition
#define TILE_SIZE 24  // N=120, 120/24=5 tiles exactly

extern "C" {

// Load a tile from global A into local tile buffer
static void load_tile(double A[N][N], double tile[TILE_SIZE][TILE_SIZE],
                      int row_start, int col_start) {
    load_tile_i: for (int i = 0; i < TILE_SIZE; i++) {
        load_tile_j: for (int j = 0; j < TILE_SIZE; j++) {
#pragma HLS PIPELINE II=1
            int gi = row_start + i;
            int gj = col_start + j;
            tile[i][j] = (gi < N && gj < N) ? A[gi][gj] : 0.0;
        }
    }
}

// Store a tile from local tile buffer back to global A
static void store_tile(double A[N][N], double tile[TILE_SIZE][TILE_SIZE],
                       int row_start, int col_start) {
    store_tile_i: for (int i = 0; i < TILE_SIZE; i++) {
        store_tile_j: for (int j = 0; j < TILE_SIZE; j++) {
#pragma HLS PIPELINE II=1
            int gi = row_start + i;
            int gj = col_start + j;
            if (gi < N && gj < N)
                A[gi][gj] = tile[i][j];
        }
    }
}

void kernel_lu(
               double A[N + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Full local copy of A for tiled computation
    double localA[N][N];
#pragma HLS ARRAY_PARTITION variable=localA cyclic factor=8 dim=2

    // ---- LOAD PHASE: bring all of A into local buffer ----
    load_i: for (int i = 0; i < N; i++) {
        load_j: for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            localA[i][j] = A[i][j];
        }
    }

    // ---- COMPUTE PHASE: blocked (tiled) LU decomposition ----
    // Number of tiles along each dimension
    const int NUM_TILES = N / TILE_SIZE;

    // Local tile buffers for the three tile roles in blocked LU
    double diag_tile[TILE_SIZE][TILE_SIZE];    // diagonal (pivot) block
    double col_panel[TILE_SIZE][TILE_SIZE];    // block in pivot column
    double row_panel[TILE_SIZE][TILE_SIZE];    // block in pivot row
    double update_tile[TILE_SIZE][TILE_SIZE];  // trailing submatrix block

#pragma HLS ARRAY_PARTITION variable=diag_tile complete dim=2
#pragma HLS ARRAY_PARTITION variable=col_panel complete dim=2
#pragma HLS ARRAY_PARTITION variable=row_panel complete dim=2
#pragma HLS ARRAY_PARTITION variable=update_tile complete dim=2

    tile_k: for (int tk = 0; tk < NUM_TILES; tk++) {
        int k0 = tk * TILE_SIZE;

        // --- Step 1: Load diagonal tile and compute its LU in-place ---
        load_diag: for (int i = 0; i < TILE_SIZE; i++) {
            for (int j = 0; j < TILE_SIZE; j++) {
#pragma HLS PIPELINE II=1
                diag_tile[i][j] = localA[k0 + i][k0 + j];
            }
        }

        // LU factorize the diagonal tile (unblocked)
        diag_lu_i: for (int i = 0; i < TILE_SIZE; i++) {
            // Lower part of diagonal tile
            diag_lu_lower_j: for (int j = 0; j < i; j++) {
                double sum = diag_tile[i][j];
                diag_lu_lower_k: for (int kk = 0; kk < j; kk++) {
#pragma HLS PIPELINE II=1
                    sum -= diag_tile[i][kk] * diag_tile[kk][j];
                }
                diag_tile[i][j] = sum / diag_tile[j][j];
            }
            // Upper part of diagonal tile
            diag_lu_upper_j: for (int j = i; j < TILE_SIZE; j++) {
                double sum = diag_tile[i][j];
                diag_lu_upper_k: for (int kk = 0; kk < i; kk++) {
#pragma HLS PIPELINE II=1
                    sum -= diag_tile[i][kk] * diag_tile[kk][j];
                }
                diag_tile[i][j] = sum;
            }
        }

        // Write factorized diagonal tile back to localA
        store_diag: for (int i = 0; i < TILE_SIZE; i++) {
            for (int j = 0; j < TILE_SIZE; j++) {
#pragma HLS PIPELINE II=1
                localA[k0 + i][k0 + j] = diag_tile[i][j];
            }
        }

        // --- Step 2: Update column panel (tiles below diagonal) ---
        col_panel_loop: for (int ti = tk + 1; ti < NUM_TILES; ti++) {
            int i0 = ti * TILE_SIZE;

            // Load this column panel tile
            load_cpanel: for (int i = 0; i < TILE_SIZE; i++) {
                for (int j = 0; j < TILE_SIZE; j++) {
#pragma HLS PIPELINE II=1
                    col_panel[i][j] = localA[i0 + i][k0 + j];
                }
            }

            // Apply L factor from diagonal tile to column panel:
            // col_panel[i][j] -= sum_kk col_panel[i][kk] * diag_tile[kk][j]  for kk < j
            // then col_panel[i][j] /= diag_tile[j][j]
            cpanel_i: for (int i = 0; i < TILE_SIZE; i++) {
                cpanel_j: for (int j = 0; j < TILE_SIZE; j++) {
                    double sum = col_panel[i][j];
                    cpanel_k: for (int kk = 0; kk < j; kk++) {
#pragma HLS PIPELINE II=1
                        sum -= col_panel[i][kk] * diag_tile[kk][j];
                    }
                    col_panel[i][j] = sum / diag_tile[j][j];
                }
            }

            // Write updated column panel back
            store_cpanel: for (int i = 0; i < TILE_SIZE; i++) {
                for (int j = 0; j < TILE_SIZE; j++) {
#pragma HLS PIPELINE II=1
                    localA[i0 + i][k0 + j] = col_panel[i][j];
                }
            }
        }

        // --- Step 3: Update row panel (tiles to the right of diagonal) ---
        row_panel_loop: for (int tj = tk + 1; tj < NUM_TILES; tj++) {
            int j0 = tj * TILE_SIZE;

            // Load row panel tile
            load_rpanel: for (int i = 0; i < TILE_SIZE; i++) {
                for (int j = 0; j < TILE_SIZE; j++) {
#pragma HLS PIPELINE II=1
                    row_panel[i][j] = localA[k0 + i][j0 + j];
                }
            }

            // Apply U factor from diagonal tile to row panel:
            // row_panel[i][j] -= sum_kk diag_tile[i][kk] * row_panel[kk][j]  for kk < i
            rpanel_i: for (int i = 0; i < TILE_SIZE; i++) {
                rpanel_j: for (int j = 0; j < TILE_SIZE; j++) {
                    double sum = row_panel[i][j];
                    rpanel_k: for (int kk = 0; kk < i; kk++) {
#pragma HLS PIPELINE II=1
                        sum -= diag_tile[i][kk] * row_panel[kk][j];
                    }
                    row_panel[i][j] = sum;
                }
            }

            // Write updated row panel back
            store_rpanel: for (int i = 0; i < TILE_SIZE; i++) {
                for (int j = 0; j < TILE_SIZE; j++) {
#pragma HLS PIPELINE II=1
                    localA[k0 + i][j0 + j] = row_panel[i][j];
                }
            }
        }

        // --- Step 4: Update trailing submatrix ---
        // For each block (ti, tj) in trailing submatrix:
        // A[ti][tj] -= col_panel[ti][k] * row_panel[k][tj]
        trail_ti: for (int ti = tk + 1; ti < NUM_TILES; ti++) {
            int i0 = ti * TILE_SIZE;

            // Reload column panel for this row block
            reload_cpanel: for (int i = 0; i < TILE_SIZE; i++) {
                for (int j = 0; j < TILE_SIZE; j++) {
#pragma HLS PIPELINE II=1
                    col_panel[i][j] = localA[i0 + i][k0 + j];
                }
            }

            trail_tj: for (int tj = tk + 1; tj < NUM_TILES; tj++) {
                int j0 = tj * TILE_SIZE;

                // Load row panel for this column block
                reload_rpanel: for (int i = 0; i < TILE_SIZE; i++) {
                    for (int j = 0; j < TILE_SIZE; j++) {
#pragma HLS PIPELINE II=1
                        row_panel[i][j] = localA[k0 + i][j0 + j];
                    }
                }

                // Load trailing tile
                load_utile: for (int i = 0; i < TILE_SIZE; i++) {
                    for (int j = 0; j < TILE_SIZE; j++) {
#pragma HLS PIPELINE II=1
                        update_tile[i][j] = localA[i0 + i][j0 + j];
                    }
                }

                // Compute rank-TILE_SIZE update: update_tile -= col_panel * row_panel
                update_i: for (int i = 0; i < TILE_SIZE; i++) {
                    update_j: for (int j = 0; j < TILE_SIZE; j++) {
                        double sum = update_tile[i][j];
                        update_k: for (int kk = 0; kk < TILE_SIZE; kk++) {
#pragma HLS PIPELINE II=1
                            sum -= col_panel[i][kk] * row_panel[kk][j];
                        }
                        update_tile[i][j] = sum;
                    }
                }

                // Store updated trailing tile back
                store_utile: for (int i = 0; i < TILE_SIZE; i++) {
                    for (int j = 0; j < TILE_SIZE; j++) {
#pragma HLS PIPELINE II=1
                        localA[i0 + i][j0 + j] = update_tile[i][j];
                    }
                }
            }
        }
    }

    // ---- STORE PHASE: write local buffer back to global A ----
    store_i: for (int i = 0; i < N; i++) {
        store_j: for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            A[i][j] = localA[i][j];
        }
    }
}

} // extern "C"