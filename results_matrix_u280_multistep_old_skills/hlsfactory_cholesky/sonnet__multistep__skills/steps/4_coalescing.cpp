#include "cholesky.h"
#include <cmath>
#include <cstring>

#ifndef TILE
#define TILE 16
#endif

extern "C" {

void kernel_cholesky(
        double A[N + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem \
    max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Full local buffer for Cholesky (holds the full working matrix)
    double L[N][N];
#pragma HLS ARRAY_PARTITION variable=L cyclic factor=8 dim=2

    // -------------------------
    // LOAD PHASE (tiled, double-buffered)
    // -------------------------
    {
        double tile_in_0[TILE][N];
        double tile_in_1[TILE][N];
#pragma HLS ARRAY_PARTITION variable=tile_in_0 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=tile_in_1 cyclic factor=8 dim=2

        const int num_tiles = (N + TILE - 1) / TILE;

        // Pre-load first tile into buffer 0
        for (int i = 0; i < TILE; i++) {
            for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
                int gi = i;
                tile_in_0[i][j] = (gi < N) ? A[gi][j] : 0.0;
            }
        }

        for (int t = 0; t < num_tiles; t++) {
#pragma HLS LOOP_TRIPCOUNT min=8 max=8
            int row_start = t * TILE;
            int flag_curr = t % 2;
            int flag_next = 1 - flag_curr;

            // Prefetch next tile
            int next_row_start = (t + 1) * TILE;
            if (next_row_start < N) {
                for (int i = 0; i < TILE; i++) {
                    for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
                        int gi = next_row_start + i;
                        double val = (gi < N) ? A[gi][j] : 0.0;
                        if (flag_next == 0)
                            tile_in_0[i][j] = val;
                        else
                            tile_in_1[i][j] = val;
                    }
                }
            }

            // Copy current tile from buffer into L
            for (int i = 0; i < TILE; i++) {
                for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS DEPENDENCE variable=L inter false
                    int gi = row_start + i;
                    if (gi < N) {
                        if (flag_curr == 0)
                            L[gi][j] = tile_in_0[i][j];
                        else
                            L[gi][j] = tile_in_1[i][j];
                    }
                }
            }
        }
    }

    // -------------------------
    // COMPUTE PHASE
    // -------------------------
    {
        double row_tile[TILE][N];
#pragma HLS ARRAY_PARTITION variable=row_tile cyclic factor=8 dim=2

        const int n = N;

        for (int tile_start = 0; tile_start < n; tile_start += TILE) {
#pragma HLS LOOP_TRIPCOUNT min=8 max=8
            int tile_end = tile_start + TILE;
            if (tile_end > n) tile_end = n;

            // Load current tile rows into local tile buffer
            for (int ti = 0; ti < TILE; ti++) {
                for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS DEPENDENCE variable=row_tile inter false
                    int gi = tile_start + ti;
                    row_tile[ti][j] = (gi < n) ? L[gi][j] : 0.0;
                }
            }

            // Process each row within this tile
            for (int ti = 0; ti < TILE; ti++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=16
                int i = tile_start + ti;
                if (i >= n) break;

                // Update row_tile[ti][j] for j < i using already-computed L rows
                for (int j = 0; j < i; j++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=119
                    double rij = row_tile[ti][j];
                    for (int k = 0; k < j; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=0 max=119
#pragma HLS DEPENDENCE variable=rij inter false
#pragma HLS DEPENDENCE variable=row_tile inter false
#pragma HLS DEPENDENCE variable=L inter false
                        rij -= row_tile[ti][k] * L[j][k];
                    }
                    row_tile[ti][j] = rij / L[j][j];
                }

                // Diagonal element
                double rii = row_tile[ti][i];
                for (int k = 0; k < i; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=0 max=119
#pragma HLS DEPENDENCE variable=row_tile inter false
                    double rik = row_tile[ti][k];
                    rii -= rik * rik;
                }
                row_tile[ti][i] = sqrt(rii);

                // Write updated row back to L immediately
                for (int j = 0; j <= i; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=1 max=120
#pragma HLS DEPENDENCE variable=L inter false
                    L[i][j] = row_tile[ti][j];
                }
                // Zero out upper triangle in L for this row
                for (int j = i + 1; j < n; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=0 max=119
#pragma HLS DEPENDENCE variable=L inter false
                    L[i][j] = 0.0;
                }
            }
        }
    }

    // -------------------------
    // STORE PHASE (tiled, double-buffered)
    // -------------------------
    {
        double tile_out_0[TILE][N];
        double tile_out_1[TILE][N];
#pragma HLS ARRAY_PARTITION variable=tile_out_0 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=tile_out_1 cyclic factor=8 dim=2

        const int num_tiles = (N + TILE - 1) / TILE;

        // Pre-fill first tile_out buffer (buffer 0) from L
        for (int i = 0; i < TILE; i++) {
            for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS DEPENDENCE variable=L inter false
                tile_out_0[i][j] = (i < N) ? L[i][j] : 0.0;
            }
        }

        for (int t = 0; t < num_tiles; t++) {
#pragma HLS LOOP_TRIPCOUNT min=8 max=8
            int row_start = t * TILE;
            int flag_curr = t % 2;
            int flag_next = 1 - flag_curr;

            // Prefetch next tile from L into the "next" buffer
            int next_row_start = (t + 1) * TILE;
            if (next_row_start < N) {
                for (int i = 0; i < TILE; i++) {
                    for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS DEPENDENCE variable=L inter false
                        int gi = next_row_start + i;
                        double val = (gi < N) ? L[gi][j] : 0.0;
                        if (flag_next == 0)
                            tile_out_0[i][j] = val;
                        else
                            tile_out_1[i][j] = val;
                    }
                }
            }

            // Store current tile to global memory
            for (int i = 0; i < TILE; i++) {
                for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
                    int gi = row_start + i;
                    if (gi < N) {
                        if (flag_curr == 0)
                            A[gi][j] = tile_out_0[i][j];
                        else
                            A[gi][j] = tile_out_1[i][j];
                    }
                }
            }
        }
    }
}

} // extern "C"