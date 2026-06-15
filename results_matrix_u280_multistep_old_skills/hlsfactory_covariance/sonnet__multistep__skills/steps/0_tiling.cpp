#include "covariance.h"
#include <string.h>

// Tile size for the N (row) dimension
#define TILE_N 16

// ---------------------------------------------------------------------------
// Load a tile of rows [row_start, row_start+TILE_N) from global data
// into local_tile[TILE_N][M]
// ---------------------------------------------------------------------------
static void load_data_tile(
    double data[N][M],
    double local_tile[TILE_N][M],
    int row_start,
    int tile_rows)
{
    for (int i = 0; i < tile_rows; i++) {
        for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
            local_tile[i][j] = data[row_start + i][j];
        }
    }
}

// ---------------------------------------------------------------------------
// Store a tile of rows back to global data
// ---------------------------------------------------------------------------
static void store_data_tile(
    double data[N][M],
    double local_tile[TILE_N][M],
    int row_start,
    int tile_rows)
{
    for (int i = 0; i < tile_rows; i++) {
        for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
            data[row_start + i][j] = local_tile[i][j];
        }
    }
}

// ---------------------------------------------------------------------------
// Compute mean: accumulate contributions from one tile of rows
// ---------------------------------------------------------------------------
static void compute_mean_tile(
    double local_tile[TILE_N][M],
    double local_mean[M],
    int tile_rows,
    bool init)
{
    for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
        if (init) local_mean[j] = 0.0;
    }
    for (int i = 0; i < tile_rows; i++) {
        for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
            local_mean[j] += local_tile[i][j];
        }
    }
}

// ---------------------------------------------------------------------------
// Center data tile: subtract mean from each element in the tile
// ---------------------------------------------------------------------------
static void compute_center_tile(
    double local_tile[TILE_N][M],
    double local_mean[M],
    int tile_rows)
{
    for (int i = 0; i < tile_rows; i++) {
        for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
            local_tile[i][j] -= local_mean[j];
        }
    }
}

// ---------------------------------------------------------------------------
// Accumulate covariance contributions from one centered tile
// partial_cov[i][j] += sum_k_in_tile( tile[k][i] * tile[k][j] )
// ---------------------------------------------------------------------------
static void compute_cov_tile(
    double local_tile[TILE_N][M],
    double partial_cov[M][M],
    int tile_rows,
    bool init)
{
    if (init) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
                partial_cov[i][j] = 0.0;
            }
        }
    }
    for (int k = 0; k < tile_rows; k++) {
        for (int i = 0; i < M; i++) {
            for (int j = i; j < M; j++) {
#pragma HLS PIPELINE II=1
                partial_cov[i][j] += local_tile[k][i] * local_tile[k][j];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Store mean to global memory
// ---------------------------------------------------------------------------
static void store_mean(double mean[M], double local_mean[M])
{
    for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
        mean[j] = local_mean[j];
    }
}

// ---------------------------------------------------------------------------
// Store covariance (symmetrize and scale) to global memory
// ---------------------------------------------------------------------------
static void store_cov(double cov[M][M], double partial_cov[M][M], double float_n)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
            double val = partial_cov[i][j] / (float_n - 1.0);
            // partial_cov was only filled for j >= i; mirror symmetrically
            cov[i][j] = (j >= i) ? val : partial_cov[j][i] / (float_n - 1.0);
        }
    }
}

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
    // Local tile buffer (one tile of rows at a time)
    // -----------------------------------------------------------------------
    double local_tile[TILE_N][M];
#pragma HLS ARRAY_PARTITION variable=local_tile cyclic factor=8 dim=2

    // Local mean and covariance accumulators (full M-sized)
    double local_mean[M];
    double local_cov[M][M];
#pragma HLS ARRAY_PARTITION variable=local_mean complete dim=1
#pragma HLS ARRAY_PARTITION variable=local_cov  cyclic factor=8 dim=2

    // -----------------------------------------------------------------------
    // PHASE 1: Compute mean — tile over rows of data
    // -----------------------------------------------------------------------
    // Initialize mean accumulator
    init_mean: for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
        local_mean[j] = 0.0;
    }

    int num_tiles = (N + TILE_N - 1) / TILE_N;

    mean_tile_loop: for (int t = 0; t < num_tiles; t++) {
        int row_start = t * TILE_N;
        int tile_rows = (row_start + TILE_N <= N) ? TILE_N : (N - row_start);

        // LOAD: bring tile into local buffer
        load_mean_i: for (int i = 0; i < tile_rows; i++) {
            load_mean_j: for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
                local_tile[i][j] = data[row_start + i][j];
            }
        }

        // COMPUTE: accumulate mean contributions from this tile
        mean_acc_i: for (int i = 0; i < tile_rows; i++) {
            mean_acc_j: for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
                local_mean[j] += local_tile[i][j];
            }
        }
    }

    // Finalize mean
    finalize_mean: for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
        local_mean[j] /= float_n;
    }

    // STORE mean
    store_mean_j: for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
        mean[j] = local_mean[j];
    }

    // -----------------------------------------------------------------------
    // PHASE 2: Center data and compute covariance — tile over rows
    // -----------------------------------------------------------------------
    // Initialize covariance accumulator
    init_cov_i: for (int i = 0; i < M; i++) {
        init_cov_j: for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
            local_cov[i][j] = 0.0;
        }
    }

    cov_tile_loop: for (int t = 0; t < num_tiles; t++) {
        int row_start = t * TILE_N;
        int tile_rows = (row_start + TILE_N <= N) ? TILE_N : (N - row_start);

        // LOAD: bring tile into local buffer
        load_cov_i: for (int i = 0; i < tile_rows; i++) {
            load_cov_j: for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
                local_tile[i][j] = data[row_start + i][j];
            }
        }

        // COMPUTE center: subtract mean from tile
        center_i: for (int i = 0; i < tile_rows; i++) {
            center_j: for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
                local_tile[i][j] -= local_mean[j];
            }
        }

        // STORE centered tile back to global memory
        store_center_i: for (int i = 0; i < tile_rows; i++) {
            store_center_j: for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
                data[row_start + i][j] = local_tile[i][j];
            }
        }

        // COMPUTE covariance accumulation from centered tile
        cov_k: for (int k = 0; k < tile_rows; k++) {
            cov_i: for (int i = 0; i < M; i++) {
                cov_j: for (int j = i; j < M; j++) {
#pragma HLS PIPELINE II=1
                    local_cov[i][j] += local_tile[k][i] * local_tile[k][j];
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // PHASE 3: Finalize and STORE covariance matrix
    // -----------------------------------------------------------------------
    store_cov_i: for (int i = 0; i < M; i++) {
        store_cov_j: for (int j = 0; j < M; j++) {
#pragma HLS PIPELINE II=1
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