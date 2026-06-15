#include "cholesky.h"
#include <cmath>
#include <cstring>

#ifndef TILE
#define TILE 16
#endif

extern "C" {

// Load a tile of rows [row_start, row_start+TILE) x [0, N) into local buffer
static void load_tile(double tile_buf[TILE][N], double A[N][N], int row_start) {
    for (int i = 0; i < TILE; i++) {
        for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            int gi = row_start + i;
            tile_buf[i][j] = (gi < N) ? A[gi][j] : 0.0;
        }
    }
}

// Store a tile of rows [row_start, row_start+TILE) x [0, N) from local buffer
static void store_tile(double A[N][N], double tile_buf[TILE][N], int row_start) {
    for (int i = 0; i < TILE; i++) {
        for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            int gi = row_start + i;
            if (gi < N) A[gi][j] = tile_buf[i][j];
        }
    }
}

void kernel_cholesky(
        double A[N + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Full local buffer for Cholesky (holds the full working matrix)
    double L[N][N];
#pragma HLS ARRAY_PARTITION variable=L cyclic factor=8 dim=2

    // -------------------------
    // LOAD PHASE (tiled)
    // -------------------------
    // Load A into L using TILE-row tiles
    {
        double tile_in[TILE][N];
#pragma HLS ARRAY_PARTITION variable=tile_in cyclic factor=8 dim=2

        for (int row_start = 0; row_start < N; row_start += TILE) {
            // Load tile from global memory
            load_tile(tile_in, A, row_start);

            // Copy tile into L
            for (int i = 0; i < TILE; i++) {
                for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
                    int gi = row_start + i;
                    if (gi < N) L[gi][j] = tile_in[i][j];
                }
            }
        }
    }

    // -------------------------
    // COMPUTE PHASE
    // -------------------------
    // Cholesky decomposition operating on local buffer L
    // Process in row tiles: for each tile of rows, use a local
    // row buffer to stage the working rows.
    {
        // Local tile buffer for the current row being updated
        double row_tile[TILE][N];
#pragma HLS ARRAY_PARTITION variable=row_tile cyclic factor=8 dim=2

        const int n = N;

        for (int tile_start = 0; tile_start < n; tile_start += TILE) {
            int tile_end = tile_start + TILE;
            if (tile_end > n) tile_end = n;

            // Load current tile rows into local tile buffer
            for (int ti = 0; ti < TILE; ti++) {
                for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
                    int gi = tile_start + ti;
                    row_tile[ti][j] = (gi < n) ? L[gi][j] : 0.0;
                }
            }

            // Process each row within this tile
            for (int ti = 0; ti < TILE; ti++) {
                int i = tile_start + ti;
                if (i >= n) break;

                // Update row_tile[ti][j] for j < i using already-computed L rows
                for (int j = 0; j < i; j++) {
                    // Accumulate: row_tile[ti][j] -= sum_k(row_tile[ti][k]*L[j][k]) for k<j
                    double rij = row_tile[ti][j];
                    for (int k = 0; k < j; k++) {
#pragma HLS PIPELINE II=1
                        rij -= row_tile[ti][k] * L[j][k];
                    }
                    row_tile[ti][j] = rij / L[j][j];
                }

                // Diagonal element
                double rii = row_tile[ti][i];
                for (int k = 0; k < i; k++) {
#pragma HLS PIPELINE II=1
                    double rik = row_tile[ti][k];
                    rii -= rik * rik;
                }
                row_tile[ti][i] = sqrt(rii);

                // Write updated row back to L immediately (other rows need it)
                for (int j = 0; j <= i; j++) {
#pragma HLS PIPELINE II=1
                    L[i][j] = row_tile[ti][j];
                }
                // Zero out upper triangle in L for this row
                for (int j = i + 1; j < n; j++) {
#pragma HLS PIPELINE II=1
                    L[i][j] = 0.0;
                }
            }
        }
    }

    // -------------------------
    // STORE PHASE (tiled)
    // -------------------------
    // Store L back to A using TILE-row tiles
    {
        double tile_out[TILE][N];
#pragma HLS ARRAY_PARTITION variable=tile_out cyclic factor=8 dim=2

        for (int row_start = 0; row_start < N; row_start += TILE) {
            // Copy tile from L into tile buffer
            for (int i = 0; i < TILE; i++) {
                for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
                    int gi = row_start + i;
                    tile_out[i][j] = (gi < N) ? L[gi][j] : 0.0;
                }
            }

            // Store tile to global memory
            store_tile(A, tile_out, row_start);
        }
    }
}

} // extern "C"