#include "gemver.h"
#include <string.h>

// Tile size: number of rows processed per tile
#define TILE 32

static void load_A_tile(
    double A[N][N],
    double l_A[TILE][N],
    int row_start, int tile_rows)
{
    for (int i = 0; i < tile_rows; i++) {
        for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            l_A[i][j] = A[row_start + i][j];
        }
    }
}

static void store_A_tile(
    double A[N][N],
    double l_A[TILE][N],
    int row_start, int tile_rows)
{
    for (int i = 0; i < tile_rows; i++) {
        for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            A[row_start + i][j] = l_A[i][j];
        }
    }
}

static void compute_loop1_tile(
    double l_A[TILE][N],
    double l_u1[N], double l_v1[N],
    double l_u2[N], double l_v2[N],
    int row_start, int tile_rows)
{
    for (int i = 0; i < tile_rows; i++) {
        double u1_i = l_u1[row_start + i];
        double u2_i = l_u2[row_start + i];
        for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            l_A[i][j] = l_A[i][j] + u1_i * l_v1[j] + u2_i * l_v2[j];
        }
    }
}

// For loop2: x[i] += beta * A[j][i] * y[j]  (transpose access)
// We accumulate partial sums into l_x for each column i,
// using a tile of rows (j dimension) from A.
static void compute_loop2_tile(
    double l_A[TILE][N],
    double l_y[N],
    double l_x_partial[N],
    double beta,
    int row_start, int tile_rows)
{
    // l_x_partial[i] accumulates beta * A[j][i] * y[j] for j in this tile
    for (int j = 0; j < tile_rows; j++) {
        int j_global = row_start + j;
        double yj = l_y[j_global];
        for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
            l_x_partial[i] += beta * l_A[j][i] * yj;
        }
    }
}

static void compute_loop4_tile(
    double l_A[TILE][N],
    double l_x[N],
    double l_w[N],
    double alpha,
    int row_start, int tile_rows)
{
    for (int i = 0; i < tile_rows; i++) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            sum += alpha * l_A[i][j] * l_x[j];
        }
        l_w[row_start + i] += sum;
    }
}

extern "C" {

void kernel_gemver(
        double alpha,
        double beta,
        double A[N + 0][N + 0],
        double u1[N + 0],
        double v1[N + 0],
        double u2[N + 0],
        double v2[N + 0],
        double w[N + 0],
        double x[N + 0],
        double y[N + 0],
        double z[N + 0])
{
#pragma HLS INTERFACE m_axi port=A    offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=u1   offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=v1   offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=u2   offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=v2   offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=w    offset=slave bundle=gmem5
#pragma HLS INTERFACE m_axi port=x    offset=slave bundle=gmem6
#pragma HLS INTERFACE m_axi port=y    offset=slave bundle=gmem7
#pragma HLS INTERFACE m_axi port=z    offset=slave bundle=gmem8
#pragma HLS INTERFACE s_axilite port=alpha  bundle=control
#pragma HLS INTERFACE s_axilite port=beta   bundle=control
#pragma HLS INTERFACE s_axilite port=A      bundle=control
#pragma HLS INTERFACE s_axilite port=u1     bundle=control
#pragma HLS INTERFACE s_axilite port=v1     bundle=control
#pragma HLS INTERFACE s_axilite port=u2     bundle=control
#pragma HLS INTERFACE s_axilite port=v2     bundle=control
#pragma HLS INTERFACE s_axilite port=w      bundle=control
#pragma HLS INTERFACE s_axilite port=x      bundle=control
#pragma HLS INTERFACE s_axilite port=y      bundle=control
#pragma HLS INTERFACE s_axilite port=z      bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // ----------------------------------------------------------------
    // Local 1D arrays (loaded once)
    // ----------------------------------------------------------------
    double l_u1[N], l_v1[N], l_u2[N], l_v2[N];
    double l_x[N], l_y[N], l_z[N], l_w[N];

#pragma HLS ARRAY_PARTITION variable=l_u1 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_v1 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_u2 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_v2 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_x  cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_y  cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_z  cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_w  cyclic factor=8 dim=1

    // ----------------------------------------------------------------
    // Tile buffer for A (TILE rows x N cols)
    // ----------------------------------------------------------------
    double l_A[TILE][N];
#pragma HLS ARRAY_PARTITION variable=l_A cyclic factor=8 dim=2

    // ----------------------------------------------------------------
    // LOAD phase: 1D arrays from global memory
    // ----------------------------------------------------------------
    load_1d: for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        l_u1[i] = u1[i];
        l_v1[i] = v1[i];
        l_u2[i] = u2[i];
        l_v2[i] = v2[i];
        l_x[i]  = x[i];
        l_y[i]  = y[i];
        l_z[i]  = z[i];
        l_w[i]  = w[i];
    }

    // ----------------------------------------------------------------
    // COMPUTE + STORE Loop1 & Loop2 (tiled over rows of A):
    //   Loop1: A[i][j] += u1[i]*v1[j] + u2[i]*v2[j]
    //   Loop2: x[i] += beta * A[j][i] * y[j]  (reuse same tile for both)
    //
    // We accumulate partial sums for loop2 into l_x_partial,
    // then add to l_x after all tiles are done.
    // ----------------------------------------------------------------
    double l_x_partial[N];
#pragma HLS ARRAY_PARTITION variable=l_x_partial cyclic factor=8 dim=1

    // Initialize partial accumulator
    init_partial: for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        l_x_partial[i] = 0.0;
    }

    // Tiled loop over rows of A
    tile_loop: for (int t = 0; t < N; t += TILE) {
        int tile_rows = ((t + TILE) <= N) ? TILE : (N - t);

        // Load tile of A from global memory
        load_A_tile(A, l_A, t, tile_rows);

        // Compute loop1 on this tile: update A[i][j]
        compute_loop1_tile(l_A, l_u1, l_v1, l_u2, l_v2, t, tile_rows);

        // Compute loop2 partial sums using the updated A tile
        // (A[j][i] where j = row index in tile, i = column)
        compute_loop2_tile(l_A, l_y, l_x_partial, beta, t, tile_rows);

        // Store updated tile of A back to global memory
        store_A_tile(A, l_A, t, tile_rows);
    }

    // Finalize loop2: add partial sums to l_x
    loop2_finalize: for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        l_x[i] += l_x_partial[i];
    }

    // ----------------------------------------------------------------
    // Loop 3: x[i] += z[i]
    // ----------------------------------------------------------------
    loop3: for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        l_x[i] = l_x[i] + l_z[i];
    }

    // ----------------------------------------------------------------
    // Loop 4 (tiled): w[i] += alpha * A[i][j] * x[j]
    // ----------------------------------------------------------------
    tile_loop4: for (int t = 0; t < N; t += TILE) {
        int tile_rows = ((t + TILE) <= N) ? TILE : (N - t);

        // Load tile of A from global memory
        load_A_tile(A, l_A, t, tile_rows);

        // Compute loop4 on this tile
        compute_loop4_tile(l_A, l_x, l_w, alpha, t, tile_rows);
    }

    // ----------------------------------------------------------------
    // STORE phase: write x and w back to global memory
    // ----------------------------------------------------------------
    store_out: for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        x[i] = l_x[i];
        w[i] = l_w[i];
    }
}

} // extern "C"