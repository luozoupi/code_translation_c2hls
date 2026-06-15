#include "lu.h"
#include <string.h>

// Tile size for blocked LU decomposition
#define TILE_SIZE 24  // N=120, 120/24=5 tiles exactly

extern "C" {

void kernel_lu(
               double A[N + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Double-buffered full local copy of A for tiled computation
    // Buffer 0 and Buffer 1 for ping-pong between load and compute
    double localA_0[N][N];
    double localA_1[N][N];
#pragma HLS ARRAY_PARTITION variable=localA_0 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=localA_1 cyclic factor=8 dim=2

    // ---- LOAD PHASE: bring all of A into buffer 0 ----
    load_i: for (int i = 0; i < N; i++) {
        load_j: for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            localA_0[i][j] = A[i][j];
            localA_1[i][j] = A[i][j];  // pre-fill both buffers identically
        }
    }

    // ---- COMPUTE PHASE: blocked (tiled) LU decomposition ----
    const int NUM_TILES = N / TILE_SIZE;

    // Local tile buffers - double buffered versions
    double diag_tile[TILE_SIZE][TILE_SIZE];

    // Double-buffered col_panel, row_panel, update_tile
    double col_panel_0[TILE_SIZE][TILE_SIZE];
    double col_panel_1[TILE_SIZE][TILE_SIZE];
    double row_panel_0[TILE_SIZE][TILE_SIZE];
    double row_panel_1[TILE_SIZE][TILE_SIZE];
    double update_tile_0[TILE_SIZE][TILE_SIZE];
    double update_tile_1[TILE_SIZE][TILE_SIZE];

#pragma HLS ARRAY_PARTITION variable=diag_tile cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=col_panel_0 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=col_panel_1 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=row_panel_0 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=row_panel_1 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=update_tile_0 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=update_tile_1 cyclic factor=8 dim=2

    tile_k: for (int tk = 0; tk < NUM_TILES; tk++) {
#pragma HLS LOOP_TRIPCOUNT min=5 max=5
        int k0 = tk * TILE_SIZE;

        // Select which localA buffer to use for this tile iteration (ping-pong)
        // Even tk -> read/write localA_0, Odd tk -> read/write localA_1
        int use_buf = tk % 2;

        // --- Step 1: Load diagonal tile ---
        load_diag: for (int i = 0; i < TILE_SIZE; i++) {
            for (int j = 0; j < TILE_SIZE; j++) {
#pragma HLS PIPELINE II=1
                if (use_buf == 0)
                    diag_tile[i][j] = localA_0[k0 + i][k0 + j];
                else
                    diag_tile[i][j] = localA_1[k0 + i][k0 + j];
            }
        }

        // LU factorize the diagonal tile (unblocked)
        diag_lu_i: for (int i = 0; i < TILE_SIZE; i++) {
            // Lower part of diagonal tile
            diag_lu_lower_j: for (int j = 0; j < i; j++) {
                double sum = diag_tile[i][j];
                diag_lu_lower_k: for (int kk = 0; kk < j; kk++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
                    sum -= diag_tile[i][kk] * diag_tile[kk][j];
                }
                diag_tile[i][j] = sum / diag_tile[j][j];
            }
            // Upper part of diagonal tile
            diag_lu_upper_j: for (int j = i; j < TILE_SIZE; j++) {
                double sum = diag_tile[i][j];
                diag_lu_upper_k: for (int kk = 0; kk < i; kk++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
                    sum -= diag_tile[i][kk] * diag_tile[kk][j];
                }
                diag_tile[i][j] = sum;
            }
        }

        // Write factorized diagonal tile back to BOTH localA buffers
        // so the next iteration (opposite buffer) has updated diagonal data
        store_diag: for (int i = 0; i < TILE_SIZE; i++) {
            for (int j = 0; j < TILE_SIZE; j++) {
#pragma HLS PIPELINE II=1
                localA_0[k0 + i][k0 + j] = diag_tile[i][j];
                localA_1[k0 + i][k0 + j] = diag_tile[i][j];
            }
        }

        // --- Step 2: Update column panel (tiles below diagonal) ---
        col_panel_loop: for (int ti = tk + 1; ti < NUM_TILES; ti++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=4
            int i0 = ti * TILE_SIZE;
            int buf_sel = ti % 2;  // double-buffer selector for col_panel

            // Load this column panel tile into appropriate buffer
            load_cpanel: for (int i = 0; i < TILE_SIZE; i++) {
                for (int j = 0; j < TILE_SIZE; j++) {
#pragma HLS PIPELINE II=1
                    double val;
                    if (use_buf == 0)
                        val = localA_0[i0 + i][k0 + j];
                    else
                        val = localA_1[i0 + i][k0 + j];
                    if (buf_sel == 0)
                        col_panel_0[i][j] = val;
                    else
                        col_panel_1[i][j] = val;
                }
            }

            // Apply L factor from diagonal tile to column panel
            cpanel_i: for (int i = 0; i < TILE_SIZE; i++) {
                cpanel_j: for (int j = 0; j < TILE_SIZE; j++) {
                    double sum;
                    if (buf_sel == 0)
                        sum = col_panel_0[i][j];
                    else
                        sum = col_panel_1[i][j];
                    cpanel_k: for (int kk = 0; kk < j; kk++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
                        if (buf_sel == 0)
                            sum -= col_panel_0[i][kk] * diag_tile[kk][j];
                        else
                            sum -= col_panel_1[i][kk] * diag_tile[kk][j];
                    }
                    if (buf_sel == 0)
                        col_panel_0[i][j] = sum / diag_tile[j][j];
                    else
                        col_panel_1[i][j] = sum / diag_tile[j][j];
                }
            }

            // Write updated column panel back to both localA buffers
            store_cpanel: for (int i = 0; i < TILE_SIZE; i++) {
                for (int j = 0; j < TILE_SIZE; j++) {
#pragma HLS PIPELINE II=1
                    double val;
                    if (buf_sel == 0)
                        val = col_panel_0[i][j];
                    else
                        val = col_panel_1[i][j];
                    localA_0[i0 + i][k0 + j] = val;
                    localA_1[i0 + i][k0 + j] = val;
                }
            }
        }

        // --- Step 3: Update row panel (tiles to the right of diagonal) ---
        row_panel_loop: for (int tj = tk + 1; tj < NUM_TILES; tj++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=4
            int j0 = tj * TILE_SIZE;
            int buf_sel = tj % 2;  // double-buffer selector for row_panel

            // Load row panel tile into appropriate buffer
            load_rpanel: for (int i = 0; i < TILE_SIZE; i++) {
                for (int j = 0; j < TILE_SIZE; j++) {
#pragma HLS PIPELINE II=1
                    double val;
                    if (use_buf == 0)
                        val = localA_0[k0 + i][j0 + j];
                    else
                        val = localA_1[k0 + i][j0 + j];
                    if (buf_sel == 0)
                        row_panel_0[i][j] = val;
                    else
                        row_panel_1[i][j] = val;
                }
            }

            // Apply U factor from diagonal tile to row panel
            rpanel_i: for (int i = 0; i < TILE_SIZE; i++) {
                rpanel_j: for (int j = 0; j < TILE_SIZE; j++) {
                    double sum;
                    if (buf_sel == 0)
                        sum = row_panel_0[i][j];
                    else
                        sum = row_panel_1[i][j];
                    rpanel_k: for (int kk = 0; kk < i; kk++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
                        if (buf_sel == 0)
                            sum -= diag_tile[i][kk] * row_panel_0[kk][j];
                        else
                            sum -= diag_tile[i][kk] * row_panel_1[kk][j];
                    }
                    if (buf_sel == 0)
                        row_panel_0[i][j] = sum;
                    else
                        row_panel_1[i][j] = sum;
                }
            }

            // Write updated row panel back to both localA buffers
            store_rpanel: for (int i = 0; i < TILE_SIZE; i++) {
                for (int j = 0; j < TILE_SIZE; j++) {
#pragma HLS PIPELINE II=1
                    double val;
                    if (buf_sel == 0)
                        val = row_panel_0[i][j];
                    else
                        val = row_panel_1[i][j];
                    localA_0[k0 + i][j0 + j] = val;
                    localA_1[k0 + i][j0 + j] = val;
                }
            }
        }

        // --- Step 4: Update trailing submatrix ---
        // Use double buffering on update_tile: load tile (ti,tj+1) while
        // computing on tile (ti,tj)
        trail_ti: for (int ti = tk + 1; ti < NUM_TILES; ti++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=4
            int i0 = ti * TILE_SIZE;
            int cp_sel = ti % 2;

            // Reload column panel for this row block into appropriate buffer
            reload_cpanel: for (int i = 0; i < TILE_SIZE; i++) {
                for (int j = 0; j < TILE_SIZE; j++) {
#pragma HLS PIPELINE II=1
                    double val;
                    if (use_buf == 0)
                        val = localA_0[i0 + i][k0 + j];
                    else
                        val = localA_1[i0 + i][k0 + j];
                    if (cp_sel == 0)
                        col_panel_0[i][j] = val;
                    else
                        col_panel_1[i][j] = val;
                }
            }

            // Double-buffer the update_tile: preload first tile, then
            // alternate load of next tile with compute of current tile
            int tj_start = tk + 1;
            int tj_end   = NUM_TILES;

            // Preload the very first row_panel and update_tile into buffer 0
            if (tj_start < tj_end) {
                int j0 = tj_start * TILE_SIZE;
                preload_rp: for (int i = 0; i < TILE_SIZE; i++) {
                    for (int j = 0; j < TILE_SIZE; j++) {
#pragma HLS PIPELINE II=1
                        double val;
                        if (use_buf == 0)
                            val = localA_0[k0 + i][j0 + j];
                        else
                            val = localA_1[k0 + i][j0 + j];
                        row_panel_0[i][j] = val;
                    }
                }
                preload_ut: for (int i = 0; i < TILE_SIZE; i++) {
                    for (int j = 0; j < TILE_SIZE; j++) {
#pragma HLS PIPELINE II=1
                        double val;
                        if (use_buf == 0)
                            val = localA_0[i0 + i][j0 + j];
                        else
                            val = localA_1[i0 + i][j0 + j];
                        update_tile_0[i][j] = val;
                    }
                }
            }

            trail_tj: for (int tj = tj_start; tj < tj_end; tj++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=4
                int j0 = tj * TILE_SIZE;
                // Current compute buffer: tj's parity
                int cur_buf = (tj - tj_start) % 2;
                int nxt_buf = 1 - cur_buf;
                int tj_next = tj + 1;

                // -- Preload NEXT tile into the opposite buffer (double buffer overlap) --
                if (tj_next < tj_end) {
                    int j0_next = tj_next * TILE_SIZE;
                    // Load next row_panel
                    load_rp_next: for (int i = 0; i < TILE_SIZE; i++) {
                        for (int j = 0; j < TILE_SIZE; j++) {
#pragma HLS PIPELINE II=1
                            double val;
                            if (use_buf == 0)
                                val = localA_0[k0 + i][j0_next + j];
                            else
                                val = localA_1[k0 + i][j0_next + j];
                            if (nxt_buf == 0)
                                row_panel_0[i][j] = val;
                            else
                                row_panel_1[i][j] = val;
                        }
                    }
                    // Load next update_tile
                    load_ut_next: for (int i = 0; i < TILE_SIZE; i++) {
                        for (int j = 0; j < TILE_SIZE; j++) {
#pragma HLS PIPELINE II=1
                            double val;
                            if (use_buf == 0)
                                val = localA_0[i0 + i][j0_next + j];
                            else
                                val = localA_1[i0 + i][j0_next + j];
                            if (nxt_buf == 0)
                                update_tile_0[i][j] = val;
                            else
                                update_tile_1[i][j] = val;
                        }
                    }
                }

                // -- Compute rank-TILE_SIZE update on current buffer --
                update_i: for (int i = 0; i < TILE_SIZE; i++) {
                    update_j: for (int j = 0; j < TILE_SIZE; j++) {
                        double sum;
                        if (cur_buf == 0)
                            sum = update_tile_0[i][j];
                        else
                            sum = update_tile_1[i][j];
                        update_k: for (int kk = 0; kk < TILE_SIZE; kk++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
                            double cp_val, rp_val;
                            if (cp_sel == 0)
                                cp_val = col_panel_0[i][kk];
                            else
                                cp_val = col_panel_1[i][kk];
                            if (cur_buf == 0)
                                rp_val = row_panel_0[kk][j];
                            else
                                rp_val = row_panel_1[kk][j];
                            sum -= cp_val * rp_val;
                        }
                        if (cur_buf == 0)
                            update_tile_0[i][j] = sum;
                        else
                            update_tile_1[i][j] = sum;
                    }
                }

                // -- Store current computed tile back to both localA buffers --
                store_utile: for (int i = 0; i < TILE_SIZE; i++) {
                    for (int j = 0; j < TILE_SIZE; j++) {
#pragma HLS PIPELINE II=1
                        double val;
                        if (cur_buf == 0)
                            val = update_tile_0[i][j];
                        else
                            val = update_tile_1[i][j];
                        localA_0[i0 + i][j0 + j] = val;
                        localA_1[i0 + i][j0 + j] = val;
                    }
                }
            } // trail_tj
        } // trail_ti
    } // tile_k

    // ---- STORE PHASE: write local buffer back to global A ----
    // Use localA_0 (both are identical after computation)
    store_i: for (int i = 0; i < N; i++) {
        store_j: for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            A[i][j] = localA_0[i][j];
        }
    }
}

} // extern "C"