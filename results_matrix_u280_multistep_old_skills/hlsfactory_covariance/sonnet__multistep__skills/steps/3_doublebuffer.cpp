#include "covariance.h"
#include <string.h>

// Tile size for the N (row) dimension
#define TILE_N 16

// ===========================================================================
// Top-level kernel
// ===========================================================================
extern "C" {

void kernel_covariance(
    double float_n,
    double data[N][M],
    double cov[M][M],
    double mean[M])
{
#pragma HLS INTERFACE m_axi port=data   offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=cov    offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=mean   offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=float_n bundle=control
#pragma HLS INTERFACE s_axilite port=data    bundle=control
#pragma HLS INTERFACE s_axilite port=cov     bundle=control
#pragma HLS INTERFACE s_axilite port=mean    bundle=control
#pragma HLS INTERFACE s_axilite port=return  bundle=control

    // -----------------------------------------------------------------------
    // Double-buffered local tile buffers (ping-pong)
    // -----------------------------------------------------------------------
    double local_tile_0[TILE_N][M];
    double local_tile_1[TILE_N][M];
#pragma HLS ARRAY_PARTITION variable=local_tile_0 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=local_tile_1 cyclic factor=8 dim=2

    // Local mean and covariance accumulators (full M-sized)
    double local_mean[M];
    double local_cov[M][M];
#pragma HLS ARRAY_PARTITION variable=local_mean complete dim=1
#pragma HLS ARRAY_PARTITION variable=local_cov  cyclic factor=8 dim=2

    int num_tiles = (N + TILE_N - 1) / TILE_N;

    // -----------------------------------------------------------------------
    // PHASE 1: Compute mean — tile over rows of data (double buffered)
    // -----------------------------------------------------------------------
    // Initialize mean accumulator
    init_mean: for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=M max=M
        local_mean[j] = 0.0;
    }

    // Pre-load first tile into buffer 0
    {
        int row_start_0 = 0;
        int tile_rows_0 = (TILE_N <= N) ? TILE_N : N;
        load_mean_pre_i: for (int i = 0; i < tile_rows_0; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_N
            load_mean_pre_j: for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=M max=M
#pragma HLS DEPENDENCE variable=local_tile_0 inter false
                local_tile_0[i][j] = data[row_start_0 + i][j];
            }
        }
    }

    mean_tile_loop: for (int t = 0; t < num_tiles; t++) {
#pragma HLS LOOP_TRIPCOUNT min=(N/TILE_N) max=(N/TILE_N+1)
        int row_start = t * TILE_N;
        int tile_rows = (row_start + TILE_N <= N) ? TILE_N : (N - row_start);

        // Determine next tile parameters
        int t_next = t + 1;
        int row_start_next = t_next * TILE_N;
        int tile_rows_next = (row_start_next + TILE_N <= N) ? TILE_N : (N - row_start_next);
        bool has_next = (t_next < num_tiles);

        int ping = t & 1; // 0 or 1

        if (ping == 0) {
            // Compute from buffer 0, load into buffer 1
            // COMPUTE: accumulate mean contributions from tile in buffer 0
            mean_acc_i_p0: for (int i = 0; i < tile_rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_N
                mean_acc_j_p0: for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=M max=M
#pragma HLS DEPENDENCE variable=local_mean inter false
                    local_mean[j] += local_tile_0[i][j];
                }
            }
            // LOAD next tile into buffer 1
            if (has_next) {
                load_mean_i_p0: for (int i = 0; i < tile_rows_next; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_N
                    load_mean_j_p0: for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=M max=M
#pragma HLS DEPENDENCE variable=local_tile_1 inter false
                        local_tile_1[i][j] = data[row_start_next + i][j];
                    }
                }
            }
        } else {
            // Compute from buffer 1, load into buffer 0
            // COMPUTE: accumulate mean contributions from tile in buffer 1
            mean_acc_i_p1: for (int i = 0; i < tile_rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_N
                mean_acc_j_p1: for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=M max=M
#pragma HLS DEPENDENCE variable=local_mean inter false
                    local_mean[j] += local_tile_1[i][j];
                }
            }
            // LOAD next tile into buffer 0
            if (has_next) {
                load_mean_i_p1: for (int i = 0; i < tile_rows_next; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_N
                    load_mean_j_p1: for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=M max=M
#pragma HLS DEPENDENCE variable=local_tile_0 inter false
                        local_tile_0[i][j] = data[row_start_next + i][j];
                    }
                }
            }
        }
    }

    // Finalize mean
    finalize_mean: for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=M max=M
        local_mean[j] /= float_n;
    }

    // STORE mean
    store_mean_j: for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=M max=M
        mean[j] = local_mean[j];
    }

    // -----------------------------------------------------------------------
    // PHASE 2: Center data and compute covariance — tile over rows (double buffered)
    // -----------------------------------------------------------------------
    // Initialize covariance accumulator
    init_cov_i: for (int i = 0; i < M; i++) {
#pragma HLS LOOP_TRIPCOUNT min=M max=M
        init_cov_j: for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=M max=M
#pragma HLS DEPENDENCE variable=local_cov inter false
            local_cov[i][j] = 0.0;
        }
    }

    // Pre-load first tile into buffer 0 for covariance phase
    {
        int row_start_0 = 0;
        int tile_rows_0 = (TILE_N <= N) ? TILE_N : N;
        load_cov_pre_i: for (int i = 0; i < tile_rows_0; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_N
            load_cov_pre_j: for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=M max=M
#pragma HLS DEPENDENCE variable=local_tile_0 inter false
                local_tile_0[i][j] = data[row_start_0 + i][j];
            }
        }
    }

    cov_tile_loop: for (int t = 0; t < num_tiles; t++) {
#pragma HLS LOOP_TRIPCOUNT min=(N/TILE_N) max=(N/TILE_N+1)
        int row_start = t * TILE_N;
        int tile_rows = (row_start + TILE_N <= N) ? TILE_N : (N - row_start);

        // Determine next tile parameters
        int t_next = t + 1;
        int row_start_next = t_next * TILE_N;
        int tile_rows_next = (row_start_next + TILE_N <= N) ? TILE_N : (N - row_start_next);
        bool has_next = (t_next < num_tiles);

        int ping = t & 1;

        if (ping == 0) {
            // Compute from buffer 0

            // CENTER: subtract mean from tile in buffer 0
            center_i_p0: for (int i = 0; i < tile_rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_N
                center_j_p0: for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=M max=M
#pragma HLS DEPENDENCE variable=local_tile_0 inter false
                    local_tile_0[i][j] -= local_mean[j];
                }
            }

            // STORE centered tile back to global memory
            store_center_i_p0: for (int i = 0; i < tile_rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_N
                store_center_j_p0: for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=M max=M
                    data[row_start + i][j] = local_tile_0[i][j];
                }
            }

            // LOAD next tile into buffer 1 while computing cov from buffer 0
            // Covariance accumulation from centered buffer 0
            cov_k_p0: for (int k = 0; k < tile_rows; k++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_N
                cov_i_p0: for (int i = 0; i < M; i++) {
#pragma HLS LOOP_TRIPCOUNT min=M max=M
                    cov_j_p0: for (int j = i; j < M; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=1 max=M
#pragma HLS DEPENDENCE variable=local_cov inter false
                        local_cov[i][j] += local_tile_0[k][i] * local_tile_0[k][j];
                    }
                }
            }

            // LOAD next tile into buffer 1
            if (has_next) {
                load_cov_i_p0: for (int i = 0; i < tile_rows_next; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_N
                    load_cov_j_p0: for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=M max=M
#pragma HLS DEPENDENCE variable=local_tile_1 inter false
                        local_tile_1[i][j] = data[row_start_next + i][j];
                    }
                }
            }

        } else {
            // Compute from buffer 1

            // CENTER: subtract mean from tile in buffer 1
            center_i_p1: for (int i = 0; i < tile_rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_N
                center_j_p1: for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=M max=M
#pragma HLS DEPENDENCE variable=local_tile_1 inter false
                    local_tile_1[i][j] -= local_mean[j];
                }
            }

            // STORE centered tile back to global memory
            store_center_i_p1: for (int i = 0; i < tile_rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_N
                store_center_j_p1: for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=M max=M
                    data[row_start + i][j] = local_tile_1[i][j];
                }
            }

            // COMPUTE covariance accumulation from centered buffer 1
            cov_k_p1: for (int k = 0; k < tile_rows; k++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_N
                cov_i_p1: for (int i = 0; i < M; i++) {
#pragma HLS LOOP_TRIPCOUNT min=M max=M
                    cov_j_p1: for (int j = i; j < M; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=1 max=M
#pragma HLS DEPENDENCE variable=local_cov inter false
                        local_cov[i][j] += local_tile_1[k][i] * local_tile_1[k][j];
                    }
                }
            }

            // LOAD next tile into buffer 0
            if (has_next) {
                load_cov_i_p1: for (int i = 0; i < tile_rows_next; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_N
                    load_cov_j_p1: for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=M max=M
#pragma HLS DEPENDENCE variable=local_tile_0 inter false
                        local_tile_0[i][j] = data[row_start_next + i][j];
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // PHASE 3: Finalize and STORE covariance matrix
    // -----------------------------------------------------------------------
    store_cov_i: for (int i = 0; i < M; i++) {
#pragma HLS LOOP_TRIPCOUNT min=M max=M
        store_cov_j: for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=M max=M
#pragma HLS DEPENDENCE variable=local_cov inter false
            double val;
            if (j >= i) {
                val = local_cov[i][j] / (float_n - 1.0);
            } else {
                val = local_cov[j][i] / (float_n - 1.0);
            }
            cov[i][j] = val;
        }
    }
}

} // extern "C"