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
#pragma HLS UNROLL factor=8
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
#pragma HLS UNROLL factor=8
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
    {
        double tile_in[TILE][N];
#pragma HLS ARRAY_PARTITION variable=tile_in cyclic factor=8 dim=2

        for (int row_start = 0; row_start < N; row_start += TILE) {
#pragma HLS LOOP_TRIPCOUNT min=8 max=8
            // Load tile from global memory
            load_tile(tile_in, A, row_start);

            // Copy tile into L
            for (int i = 0; i < TILE; i++) {
                for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS DEPENDENCE variable=L inter false
                    int gi = row_start + i;
                    if (gi < N) L[gi][j] = tile_in[i][j];
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
                    // Accumulate: row_tile[ti][j] -= sum_k(row_tile[ti][k]*L[j][k]) for k<j
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

                // Write updated row back to L immediately (other rows need it)
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
    // STORE PHASE (tiled)
    // -------------------------
    {
        double tile_out[TILE][N];
#pragma HLS ARRAY_PARTITION variable=tile_out cyclic factor=8 dim=2

        for (int row_start = 0; row_start < N; row_start += TILE) {
#pragma HLS LOOP_TRIPCOUNT min=8 max=8
            // Copy tile from L into tile buffer
            for (int i = 0; i < TILE; i++) {
                for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS DEPENDENCE variable=L inter false
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