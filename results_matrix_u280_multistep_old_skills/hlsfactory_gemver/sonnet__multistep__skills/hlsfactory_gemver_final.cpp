#include "gemver.h"
#include <string.h>

// Tile size: number of rows processed per tile
#define TILE 32

static void load_A_tile(
    double l_A_0[TILE][N],
    double l_A_1[TILE][N],
    int buf_sel,
    int row_start, int tile_rows,
    double flat_A[N*N])
{
    if (buf_sel == 0) {
        for (int i = 0; i < tile_rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
            for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=N max=N
                l_A_0[i][j] = flat_A[(row_start + i) * N + j];
            }
        }
    } else {
        for (int i = 0; i < tile_rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
            for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=N max=N
                l_A_1[i][j] = flat_A[(row_start + i) * N + j];
            }
        }
    }
}

static void store_A_tile(
    double flat_A[N*N],
    double l_A_0[TILE][N],
    double l_A_1[TILE][N],
    int buf_sel,
    int row_start, int tile_rows)
{
    if (buf_sel == 0) {
        for (int i = 0; i < tile_rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
            for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=N max=N
                flat_A[(row_start + i) * N + j] = l_A_0[i][j];
            }
        }
    } else {
        for (int i = 0; i < tile_rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
            for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=N max=N
                flat_A[(row_start + i) * N + j] = l_A_1[i][j];
            }
        }
    }
}

static void compute_loop1_tile(
    double l_A_0[TILE][N],
    double l_A_1[TILE][N],
    int buf_sel,
    double l_u1[N], double l_v1[N],
    double l_u2[N], double l_v2[N],
    int row_start, int tile_rows)
{
#pragma HLS ARRAY_PARTITION variable=l_A_0 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_A_1 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_v1  cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_v2  cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_u1  cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_u2  cyclic factor=8 dim=1

    if (buf_sel == 0) {
        for (int i = 0; i < tile_rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
            double u1_i = l_u1[row_start + i];
            double u2_i = l_u2[row_start + i];
            for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=N max=N
#pragma HLS DEPENDENCE variable=l_A_0 inter false
                l_A_0[i][j] = l_A_0[i][j] + u1_i * l_v1[j] + u2_i * l_v2[j];
            }
        }
    } else {
        for (int i = 0; i < tile_rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
            double u1_i = l_u1[row_start + i];
            double u2_i = l_u2[row_start + i];
            for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=N max=N
#pragma HLS DEPENDENCE variable=l_A_1 inter false
                l_A_1[i][j] = l_A_1[i][j] + u1_i * l_v1[j] + u2_i * l_v2[j];
            }
        }
    }
}

static void compute_loop2_tile(
    double l_A_0[TILE][N],
    double l_A_1[TILE][N],
    int buf_sel,
    double l_y[N],
    double l_x_partial[N],
    double beta,
    int row_start, int tile_rows)
{
#pragma HLS ARRAY_PARTITION variable=l_A_0        cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_A_1        cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_x_partial  cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_y          cyclic factor=8 dim=1

    if (buf_sel == 0) {
        for (int j = 0; j < tile_rows; j++) {
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
            int j_global = row_start + j;
            double yj = l_y[j_global];
            for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=N max=N
#pragma HLS DEPENDENCE variable=l_x_partial inter false
                l_x_partial[i] += beta * l_A_0[j][i] * yj;
            }
        }
    } else {
        for (int j = 0; j < tile_rows; j++) {
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
            int j_global = row_start + j;
            double yj = l_y[j_global];
            for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=N max=N
#pragma HLS DEPENDENCE variable=l_x_partial inter false
                l_x_partial[i] += beta * l_A_1[j][i] * yj;
            }
        }
    }
}

static void compute_loop4_tile(
    double l_A_0[TILE][N],
    double l_A_1[TILE][N],
    int buf_sel,
    double l_x[N],
    double l_w[N],
    double alpha,
    int row_start, int tile_rows)
{
#pragma HLS ARRAY_PARTITION variable=l_A_0 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_A_1 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_x   cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_w   cyclic factor=8 dim=1

    if (buf_sel == 0) {
        for (int i = 0; i < tile_rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
            double sum = 0.0;
            for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=N max=N
#pragma HLS DEPENDENCE variable=l_A_0 inter false
                sum += alpha * l_A_0[i][j] * l_x[j];
            }
            l_w[row_start + i] += sum;
        }
    } else {
        for (int i = 0; i < tile_rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
            double sum = 0.0;
            for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=N max=N
#pragma HLS DEPENDENCE variable=l_A_1 inter false
                sum += alpha * l_A_1[i][j] * l_x[j];
            }
            l_w[row_start + i] += sum;
        }
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
#pragma HLS INTERFACE m_axi port=A    offset=slave bundle=gmem0 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=u1   offset=slave bundle=gmem1 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=v1   offset=slave bundle=gmem2 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=u2   offset=slave bundle=gmem3 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=v2   offset=slave bundle=gmem4 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=w    offset=slave bundle=gmem5 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=x    offset=slave bundle=gmem6 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=y    offset=slave bundle=gmem7 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=z    offset=slave bundle=gmem8 max_read_burst_length=256 max_write_burst_length=256
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
    // Double-buffered tile buffers for A (ping-pong)
    // ----------------------------------------------------------------
    double l_A_0[TILE][N];
    double l_A_1[TILE][N];
#pragma HLS ARRAY_PARTITION variable=l_A_0 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_A_1 cyclic factor=8 dim=2

    // Flat local buffer for A to enable coalesced burst transfers
    double flat_A[N * N];
#pragma HLS ARRAY_PARTITION variable=flat_A cyclic factor=8 dim=1

    // ----------------------------------------------------------------
    // LOAD phase: 1D arrays from global memory (pipelined burst)
    // ----------------------------------------------------------------
    load_1d: for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=N max=N
        l_u1[i] = u1[i];
        l_v1[i] = v1[i];
        l_u2[i] = u2[i];
        l_v2[i] = v2[i];
        l_x[i]  = x[i];
        l_y[i]  = y[i];
        l_z[i]  = z[i];
        l_w[i]  = w[i];
    }

    // Load entire A matrix into flat_A for coalesced access
    load_A_flat: for (int i = 0; i < N; i++) {
#pragma HLS LOOP_TRIPCOUNT min=N max=N
        for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=N max=N
            flat_A[i * N + j] = A[i][j];
        }
    }

    // ----------------------------------------------------------------
    // Partial accumulator for loop2
    // ----------------------------------------------------------------
    double l_x_partial[N];
#pragma HLS ARRAY_PARTITION variable=l_x_partial cyclic factor=8 dim=1

    init_partial: for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=N max=N
        l_x_partial[i] = 0.0;
    }

    // ----------------------------------------------------------------
    // DOUBLE-BUFFERED tiled loop over rows of A (Loop1 + Loop2)
    // ----------------------------------------------------------------
    const int num_tiles = (N + TILE - 1) / TILE;

    // Preload tile 0 into buffer 0
    {
        int t0 = 0;
        int tile_rows0 = ((t0 + TILE) <= N) ? TILE : (N - t0);
        load_A_tile(l_A_0, l_A_1, 0, t0, tile_rows0, flat_A);
    }

    tile_loop: for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
#pragma HLS LOOP_TRIPCOUNT min=N/TILE max=N/TILE
        int t_cur  = tile_idx * TILE;
        int t_next = (tile_idx + 1) * TILE;

        int tile_rows_cur  = ((t_cur  + TILE) <= N) ? TILE : (N - t_cur);
        int tile_rows_next = ((t_next + TILE) <= N) ? TILE : (N - t_next);

        int buf_cur  = tile_idx % 2;
        int buf_next = 1 - buf_cur;

        // Compute on current tile
        compute_loop1_tile(l_A_0, l_A_1, buf_cur,
                           l_u1, l_v1, l_u2, l_v2,
                           t_cur, tile_rows_cur);

        compute_loop2_tile(l_A_0, l_A_1, buf_cur,
                           l_y, l_x_partial, beta,
                           t_cur, tile_rows_cur);

        // Store current tile back to flat_A
        store_A_tile(flat_A, l_A_0, l_A_1, buf_cur, t_cur, tile_rows_cur);

        // Load next tile into buf_next
        if (tile_idx + 1 < num_tiles) {
            load_A_tile(l_A_0, l_A_1, buf_next, t_next, tile_rows_next, flat_A);
        }
    }

    // Finalize loop2: add partial sums to l_x
    loop2_finalize: for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=N max=N
#pragma HLS DEPENDENCE variable=l_x inter false
        l_x[i] += l_x_partial[i];
    }

    // ----------------------------------------------------------------
    // Loop 3: x[i] += z[i]
    // ----------------------------------------------------------------
    loop3: for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=N max=N
#pragma HLS DEPENDENCE variable=l_x inter false
        l_x[i] = l_x[i] + l_z[i];
    }

    // ----------------------------------------------------------------
    // DOUBLE-BUFFERED Loop 4 (tiled): w[i] += alpha * A[i][j] * x[j]
    // ----------------------------------------------------------------

    // Preload tile 0 into buffer 0 for loop4
    {
        int t0 = 0;
        int tile_rows0 = ((t0 + TILE) <= N) ? TILE : (N - t0);
        load_A_tile(l_A_0, l_A_1, 0, t0, tile_rows0, flat_A);
    }

    tile_loop4: for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
#pragma HLS LOOP_TRIPCOUNT min=N/TILE max=N/TILE
        int t_cur  = tile_idx * TILE;
        int t_next = (tile_idx + 1) * TILE;

        int tile_rows_cur  = ((t_cur  + TILE) <= N) ? TILE : (N - t_cur);
        int tile_rows_next = ((t_next + TILE) <= N) ? TILE : (N - t_next);

        int buf_cur  = tile_idx % 2;
        int buf_next = 1 - buf_cur;

        // Compute loop4 on current tile
        compute_loop4_tile(l_A_0, l_A_1, buf_cur,
                           l_x, l_w, alpha,
                           t_cur, tile_rows_cur);

        // Load next tile into buf_next
        if (tile_idx + 1 < num_tiles) {
            load_A_tile(l_A_0, l_A_1, buf_next, t_next, tile_rows_next, flat_A);
        }
    }

    // ----------------------------------------------------------------
    // STORE phase: write x, w, and A back to global memory
    // ----------------------------------------------------------------
    store_out: for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=N max=N
        x[i] = l_x[i];
        w[i] = l_w[i];
    }

    // Write updated A back from flat_A to global memory (coalesced burst)
    store_A_flat: for (int i = 0; i < N; i++) {
#pragma HLS LOOP_TRIPCOUNT min=N max=N
        for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=N max=N
            A[i][j] = flat_A[i * N + j];
        }
    }
}

} // extern "C"