#include "correlation.h"
#include <cmath>
#include <string.h>

// Tile sizes
#define TILE_J 16   // tile size along M (column) dimension
#define TILE_I 16   // tile size along N (row) dimension

// Unroll factor for inner loops
#define UNROLL_J 8

extern "C" {

void kernel_correlation(
            double float_n,
            double data[N][M],
            double corr[M][M],
            double mean[M],
            double stddev[M])
{
#pragma HLS INTERFACE m_axi port=data   offset=slave bundle=gmem  max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=corr   offset=slave bundle=gmem1 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=mean   offset=slave bundle=gmem2 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=stddev offset=slave bundle=gmem3 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE s_axilite port=float_n bundle=control
#pragma HLS INTERFACE s_axilite port=data    bundle=control
#pragma HLS INTERFACE s_axilite port=corr    bundle=control
#pragma HLS INTERFACE s_axilite port=mean    bundle=control
#pragma HLS INTERFACE s_axilite port=stddev  bundle=control
#pragma HLS INTERFACE s_axilite port=return  bundle=control

    // -------------------------------------------------------
    // Full local arrays for intermediate computation
    // -------------------------------------------------------
    double l_data   [N][M];
    double l_corr   [M][M];
    double l_mean   [M];
    double l_stddev [M];

    // Partition with factor=8 to match bus width (8 doubles per 512-bit word)
#pragma HLS ARRAY_PARTITION variable=l_data    cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_corr    cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_mean    complete dim=1
#pragma HLS ARRAY_PARTITION variable=l_stddev  complete dim=1

    // Tile buffer for load/store phases - partition to match unroll
    double tile_buf[TILE_I][TILE_J];
#pragma HLS ARRAY_PARTITION variable=tile_buf complete dim=2

    const int n = N;
    const int m = M;
    double eps = 0.1;

    // -------------------------------------------------------
    // LOAD PHASE: Load data from global memory in tiles
    // -------------------------------------------------------
    load_tile_i: for (int i0 = 0; i0 < n; i0 += TILE_I) {
#pragma HLS LOOP_TRIPCOUNT min=7 max=7
        load_tile_j: for (int j0 = 0; j0 < m; j0 += TILE_J) {
#pragma HLS LOOP_TRIPCOUNT min=5 max=5
            // Load one tile from global memory into tile_buf
            load_gmem_i: for (int ti = 0; ti < TILE_I; ti++) {
                load_gmem_j: for (int tj = 0; tj < TILE_J; tj++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS DEPENDENCE variable=tile_buf inter false
                    int gi = i0 + ti;
                    int gj = j0 + tj;
                    if (gi < n && gj < m)
                        tile_buf[ti][tj] = data[gi][gj];
                    else
                        tile_buf[ti][tj] = 0.0;
                }
            }
            // Copy tile_buf into l_data
            copy_tile_i: for (int ti = 0; ti < TILE_I; ti++) {
                copy_tile_j: for (int tj = 0; tj < TILE_J; tj++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS DEPENDENCE variable=l_data inter false
                    int gi = i0 + ti;
                    int gj = j0 + tj;
                    if (gi < n && gj < m)
                        l_data[gi][gj] = tile_buf[ti][tj];
                }
            }
        }
    }

    // -------------------------------------------------------
    // COMPUTE PHASE 1: Compute mean (tiled over columns)
    // -------------------------------------------------------
    mean_tile_j: for (int j0 = 0; j0 < m; j0 += TILE_J) {
#pragma HLS LOOP_TRIPCOUNT min=5 max=5
        double tile_sum[TILE_J];
#pragma HLS ARRAY_PARTITION variable=tile_sum complete dim=1

        // Initialize accumulators
        init_mean: for (int tj = 0; tj < TILE_J; tj++) {
#pragma HLS UNROLL
            tile_sum[tj] = 0.0;
        }

        // Accumulate over rows, processing TILE_J columns at a time
        mean_tile_i: for (int i = 0; i < n; i++) {
#pragma HLS LOOP_TRIPCOUNT min=100 max=100
            mean_tile_jj: for (int tj = 0; tj < TILE_J; tj++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS DEPENDENCE variable=tile_sum inter false
#pragma HLS DEPENDENCE variable=l_data inter false
                int gj = j0 + tj;
                if (gj < m)
                    tile_sum[tj] += l_data[i][gj];
            }
        }

        // Store means for this column tile
        store_mean_tile: for (int tj = 0; tj < TILE_J; tj++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
            int gj = j0 + tj;
            if (gj < m)
                l_mean[gj] = tile_sum[tj] / float_n;
        }
    }

    // -------------------------------------------------------
    // COMPUTE PHASE 2: Compute stddev (tiled over columns)
    // -------------------------------------------------------
    stddev_tile_j: for (int j0 = 0; j0 < m; j0 += TILE_J) {
#pragma HLS LOOP_TRIPCOUNT min=5 max=5
        double tile_sum[TILE_J];
#pragma HLS ARRAY_PARTITION variable=tile_sum complete dim=1

        // Initialize
        init_stddev: for (int tj = 0; tj < TILE_J; tj++) {
#pragma HLS UNROLL
            tile_sum[tj] = 0.0;
        }

        // Accumulate squared differences
        stddev_tile_i: for (int i = 0; i < n; i++) {
#pragma HLS LOOP_TRIPCOUNT min=100 max=100
            stddev_tile_jj: for (int tj = 0; tj < TILE_J; tj++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS DEPENDENCE variable=tile_sum inter false
#pragma HLS DEPENDENCE variable=l_data inter false
#pragma HLS DEPENDENCE variable=l_mean inter false
                int gj = j0 + tj;
                if (gj < m) {
                    double diff = l_data[i][gj] - l_mean[gj];
                    tile_sum[tj] += diff * diff;
                }
            }
        }

        // Compute and store stddev for this column tile
        store_stddev_tile: for (int tj = 0; tj < TILE_J; tj++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
            int gj = j0 + tj;
            if (gj < m) {
                double s = tile_sum[tj] / float_n;
                s = sqrt(s);
                l_stddev[gj] = (s <= eps) ? 1.0 : s;
            }
        }
    }

    // -------------------------------------------------------
    // COMPUTE PHASE 3: Normalize data (tiled over rows x cols)
    // -------------------------------------------------------
    double sqrt_float_n = sqrt(float_n);

    norm_tile_i: for (int i0 = 0; i0 < n; i0 += TILE_I) {
#pragma HLS LOOP_TRIPCOUNT min=7 max=7
        norm_tile_j: for (int j0 = 0; j0 < m; j0 += TILE_J) {
#pragma HLS LOOP_TRIPCOUNT min=5 max=5
            // Load tile into tile_buf
            norm_load_i: for (int ti = 0; ti < TILE_I; ti++) {
                norm_load_j: for (int tj = 0; tj < TILE_J; tj++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS DEPENDENCE variable=tile_buf inter false
#pragma HLS DEPENDENCE variable=l_data inter false
                    int gi = i0 + ti;
                    int gj = j0 + tj;
                    if (gi < n && gj < m)
                        tile_buf[ti][tj] = l_data[gi][gj];
                    else
                        tile_buf[ti][tj] = 0.0;
                }
            }

            // Normalize within tile
            norm_compute_i: for (int ti = 0; ti < TILE_I; ti++) {
                norm_compute_j: for (int tj = 0; tj < TILE_J; tj++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS DEPENDENCE variable=tile_buf inter false
#pragma HLS DEPENDENCE variable=l_stddev inter false
                    int gi = i0 + ti;
                    int gj = j0 + tj;
                    if (gi < n && gj < m)
                        tile_buf[ti][tj] = tile_buf[ti][tj] / (sqrt_float_n * l_stddev[gj]);
                }
            }

            // Write normalized tile back to l_data
            norm_store_i: for (int ti = 0; ti < TILE_I; ti++) {
                norm_store_j: for (int tj = 0; tj < TILE_J; tj++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS DEPENDENCE variable=l_data inter false
#pragma HLS DEPENDENCE variable=tile_buf inter false
                    int gi = i0 + ti;
                    int gj = j0 + tj;
                    if (gi < n && gj < m)
                        l_data[gi][gj] = tile_buf[ti][tj];
                }
            }
        }
    }

    // -------------------------------------------------------
    // COMPUTE PHASE 4: Correlation matrix (tiled over i,j pairs)
    // -------------------------------------------------------
    corr_tile_i: for (int i = 0; i < m - 1; i++) {
#pragma HLS LOOP_TRIPCOUNT min=79 max=79
        l_corr[i][i] = 1.0;

        // Tile j loop in chunks of TILE_J
        corr_tile_j: for (int j0 = i + 1; j0 < m; j0 += TILE_J) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=5
            // Accumulator for each j in the tile
            double tile_acc[TILE_J];
#pragma HLS ARRAY_PARTITION variable=tile_acc complete dim=1

            // Initialize accumulators
            init_corr: for (int tj = 0; tj < TILE_J; tj++) {
#pragma HLS UNROLL
                tile_acc[tj] = 0.0;
            }

            // Accumulate over rows k
            corr_k: for (int k = 0; k < n; k++) {
#pragma HLS LOOP_TRIPCOUNT min=100 max=100
                double dki = l_data[k][i];
                corr_jj: for (int tj = 0; tj < TILE_J; tj++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS DEPENDENCE variable=tile_acc inter false
#pragma HLS DEPENDENCE variable=l_data inter false
                    int gj = j0 + tj;
                    if (gj < m)
                        tile_acc[tj] += dki * l_data[k][gj];
                }
            }

            // Store tile results
            store_corr_tile: for (int tj = 0; tj < TILE_J; tj++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS DEPENDENCE variable=l_corr inter false
                int gj = j0 + tj;
                if (gj < m) {
                    l_corr[i][gj] = tile_acc[tj];
                    l_corr[gj][i] = tile_acc[tj];
                }
            }
        }
    }
    l_corr[m-1][m-1] = 1.0;

    // -------------------------------------------------------
    // STORE PHASE: Write results back to global memory
    // -------------------------------------------------------

    // Store mean and stddev
    store_mean_j: for (int j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=80 max=80
        mean[j]   = l_mean[j];
        stddev[j] = l_stddev[j];
    }

    // Store correlation matrix in tiles
    store_corr_tile_i: for (int i0 = 0; i0 < m; i0 += TILE_J) {
#pragma HLS LOOP_TRIPCOUNT min=5 max=5
        store_corr_tile_j: for (int j0 = 0; j0 < m; j0 += TILE_J) {
#pragma HLS LOOP_TRIPCOUNT min=5 max=5
            // Load corr tile into tile_buf (reuse, note TILE_J x TILE_J)
            sc_load_i: for (int ti = 0; ti < TILE_J; ti++) {
                sc_load_j: for (int tj = 0; tj < TILE_J; tj++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS DEPENDENCE variable=tile_buf inter false
#pragma HLS DEPENDENCE variable=l_corr inter false
                    int gi = i0 + ti;
                    int gj = j0 + tj;
                    if (gi < m && gj < m)
                        tile_buf[ti][tj] = l_corr[gi][gj];
                    else
                        tile_buf[ti][tj] = 0.0;
                }
            }
            // Write tile to global memory
            sc_store_i: for (int ti = 0; ti < TILE_J; ti++) {
                sc_store_j: for (int tj = 0; tj < TILE_J; tj++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS DEPENDENCE variable=tile_buf inter false
                    int gi = i0 + ti;
                    int gj = j0 + tj;
                    if (gi < m && gj < m)
                        corr[gi][gj] = tile_buf[ti][tj];
                }
            }
        }
    }

    // Store normalized data in tiles
    store_data_tile_i: for (int i0 = 0; i0 < n; i0 += TILE_I) {
#pragma HLS LOOP_TRIPCOUNT min=7 max=7
        store_data_tile_j: for (int j0 = 0; j0 < m; j0 += TILE_J) {
#pragma HLS LOOP_TRIPCOUNT min=5 max=5
            // Load l_data tile into tile_buf
            sd_load_i: for (int ti = 0; ti < TILE_I; ti++) {
                sd_load_j: for (int tj = 0; tj < TILE_J; tj++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS DEPENDENCE variable=tile_buf inter false
#pragma HLS DEPENDENCE variable=l_data inter false
                    int gi = i0 + ti;
                    int gj = j0 + tj;
                    if (gi < n && gj < m)
                        tile_buf[ti][tj] = l_data[gi][gj];
                    else
                        tile_buf[ti][tj] = 0.0;
                }
            }
            // Write tile to global memory
            sd_store_i: for (int ti = 0; ti < TILE_I; ti++) {
                sd_store_j: for (int tj = 0; tj < TILE_J; tj++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS DEPENDENCE variable=tile_buf inter false
                    int gi = i0 + ti;
                    int gj = j0 + tj;
                    if (gi < n && gj < m)
                        data[gi][gj] = tile_buf[ti][tj];
                }
            }
        }
    }
}

} // extern "C"