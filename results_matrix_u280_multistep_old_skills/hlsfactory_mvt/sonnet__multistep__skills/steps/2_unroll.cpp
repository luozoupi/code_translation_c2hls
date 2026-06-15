#include "mvt.h"
#include <string.h>

#ifndef TILE
#define TILE 32
#endif

extern "C" {

// Load a tile of y_1 into local buffer
static void load_y1_tile(double y_1[N], double local_y1[TILE], int j_start, int tile_size) {
    for (int j = 0; j < tile_size; j++) {
#pragma HLS PIPELINE II=1
        local_y1[j] = y_1[j_start + j];
    }
}

// Load a tile of y_2 into local buffer
static void load_y2_tile(double y_2[N], double local_y2[TILE], int j_start, int tile_size) {
    for (int j = 0; j < tile_size; j++) {
#pragma HLS PIPELINE II=1
        local_y2[j] = y_2[j_start + j];
    }
}

// Load a tile of A rows [i_start..i_start+tile_rows) x columns [j_start..j_start+tile_cols)
static void load_A_tile(double A[N][N], double local_A[TILE][TILE],
                        int i_start, int j_start,
                        int tile_rows, int tile_cols) {
    for (int i = 0; i < tile_rows; i++) {
        for (int j = 0; j < tile_cols; j++) {
#pragma HLS PIPELINE II=1
            local_A[i][j] = A[i_start + i][j_start + j];
        }
    }
}

// Load a tile of A transposed: rows [j_start..j_start+tile_rows) x columns [i_start..i_start+tile_cols)
// For loop2: A[j][i], so we load A[j_start..j_start+TILE][i_start..i_start+TILE]
static void load_A_tile_T(double A[N][N], double local_A[TILE][TILE],
                           int j_start, int i_start,
                           int tile_rows, int tile_cols) {
    for (int j = 0; j < tile_rows; j++) {
        for (int i = 0; i < tile_cols; i++) {
#pragma HLS PIPELINE II=1
            local_A[j][i] = A[j_start + j][i_start + i];
        }
    }
}

void kernel_mvt(
		double x1[ N + 0],
		double x2[ N + 0],
		double y_1[ N + 0],
		double y_2[ N + 0],
		double A[ N + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=x1  offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=x2  offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=y_1 offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=y_2 offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=A   offset=slave bundle=gmem4
#pragma HLS INTERFACE s_axilite port=x1      bundle=control
#pragma HLS INTERFACE s_axilite port=x2      bundle=control
#pragma HLS INTERFACE s_axilite port=y_1     bundle=control
#pragma HLS INTERFACE s_axilite port=y_2     bundle=control
#pragma HLS INTERFACE s_axilite port=A       bundle=control
#pragma HLS INTERFACE s_axilite port=return  bundle=control

    const int n = N;

    // -------------------------------------------------------
    // Loop 1: x1[i] += A[i][j] * y_1[j]
    // Tile over i (rows of A) and j (columns of A / y_1)
    // -------------------------------------------------------
    loop1_i_tile: for (int i_start = 0; i_start < n; i_start += TILE) {
#pragma HLS LOOP_TRIPCOUNT min=(N+TILE-1)/TILE max=(N+TILE-1)/TILE
        int i_tile = (i_start + TILE <= n) ? TILE : (n - i_start);

        // Local buffer for x1 tile (accumulator)
        double local_x1[TILE];
#pragma HLS ARRAY_PARTITION variable=local_x1 complete dim=1

        // Load x1 tile from global memory
        load_x1: for (int i = 0; i < i_tile; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
            local_x1[i] = x1[i_start + i];
        }

        loop1_j_tile: for (int j_start = 0; j_start < n; j_start += TILE) {
#pragma HLS LOOP_TRIPCOUNT min=(N+TILE-1)/TILE max=(N+TILE-1)/TILE
            int j_tile = (j_start + TILE <= n) ? TILE : (n - j_start);

            // Local buffer for y_1 tile
            double local_y1[TILE];
#pragma HLS ARRAY_PARTITION variable=local_y1 complete dim=1

            // Local buffer for A tile [i_tile x j_tile]
            // Partition dim=2 (j-dimension) with factor=4 to match unroll factor
            double local_A1[TILE][TILE];
#pragma HLS ARRAY_PARTITION variable=local_A1 complete dim=2

            // Load phase: y_1 tile
            load_y1_ph: for (int j = 0; j < j_tile; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
                local_y1[j] = y_1[j_start + j];
            }

            // Load phase: A tile
            load_A1_ph: for (int i = 0; i < i_tile; i++) {
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
                for (int j = 0; j < j_tile; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
                    local_A1[i][j] = A[i_start + i][j_start + j];
                }
            }

            // Compute phase: accumulate into local_x1
            compute1_i: for (int i = 0; i < i_tile; i++) {
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
#pragma HLS DEPENDENCE variable=local_x1 inter false
                double sum = local_x1[i];
                compute1_j: for (int j = 0; j < j_tile; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
#pragma HLS DEPENDENCE variable=local_A1 inter false
#pragma HLS DEPENDENCE variable=local_y1 inter false
                    sum += local_A1[i][j] * local_y1[j];
                }
                local_x1[i] = sum;
            }
        }

        // Store phase: write x1 tile back to global memory
        store_x1: for (int i = 0; i < i_tile; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
            x1[i_start + i] = local_x1[i];
        }
    }

    // -------------------------------------------------------
    // Loop 2: x2[i] += A[j][i] * y_2[j]
    // Tile over i (columns of A) and j (rows of A / y_2)
    // -------------------------------------------------------
    loop2_i_tile: for (int i_start = 0; i_start < n; i_start += TILE) {
#pragma HLS LOOP_TRIPCOUNT min=(N+TILE-1)/TILE max=(N+TILE-1)/TILE
        int i_tile = (i_start + TILE <= n) ? TILE : (n - i_start);

        // Local buffer for x2 tile (accumulator)
        double local_x2[TILE];
#pragma HLS ARRAY_PARTITION variable=local_x2 complete dim=1

        // Load x2 tile from global memory
        load_x2: for (int i = 0; i < i_tile; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
            local_x2[i] = x2[i_start + i];
        }

        loop2_j_tile: for (int j_start = 0; j_start < n; j_start += TILE) {
#pragma HLS LOOP_TRIPCOUNT min=(N+TILE-1)/TILE max=(N+TILE-1)/TILE
            int j_tile = (j_start + TILE <= n) ? TILE : (n - j_start);

            // Local buffer for y_2 tile
            double local_y2[TILE];
#pragma HLS ARRAY_PARTITION variable=local_y2 complete dim=1

            // Local buffer for A tile: A[j_start..j_start+j_tile][i_start..i_start+i_tile]
            // Store as local_A2[j][i] to match A[j][i] access pattern
            // Partition dim=1 (j-dimension) with factor=4 to match unroll factor
            double local_A2[TILE][TILE];
#pragma HLS ARRAY_PARTITION variable=local_A2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=local_A2 complete dim=2

            // Load phase: y_2 tile
            load_y2_ph: for (int j = 0; j < j_tile; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
                local_y2[j] = y_2[j_start + j];
            }

            // Load phase: A tile (loading A[j][i] block, row by row for burst)
            load_A2_ph: for (int j = 0; j < j_tile; j++) {
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
                for (int i = 0; i < i_tile; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
                    local_A2[j][i] = A[j_start + j][i_start + i];
                }
            }

            // Compute phase: accumulate into local_x2
            // x2[i] += sum_j A[j][i] * y_2[j]
            // = sum_j local_A2[j][i] * local_y2[j]
            compute2_i: for (int i = 0; i < i_tile; i++) {
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
#pragma HLS DEPENDENCE variable=local_x2 inter false
                double sum = local_x2[i];
                compute2_j: for (int j = 0; j < j_tile; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
#pragma HLS DEPENDENCE variable=local_A2 inter false
#pragma HLS DEPENDENCE variable=local_y2 inter false
                    sum += local_A2[j][i] * local_y2[j];
                }
                local_x2[i] = sum;
            }
        }

        // Store phase: write x2 tile back to global memory
        store_x2: for (int i = 0; i < i_tile; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
            x2[i_start + i] = local_x2[i];
        }
    }
}

} // extern "C"