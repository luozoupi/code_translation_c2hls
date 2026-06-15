#include "syr2k.h"
#include <string.h>

// Tile sizes
#define TILE_N 16
#define TILE_M 16

static void load_tile_C(
    double C[N][N],
    double tileC[TILE_N][TILE_N],
    int i_base, int j_base,
    int tile_rows, int tile_cols)
{
    for (int i = 0; i < tile_rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_N
        for (int j = 0; j < tile_cols; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_N
            tileC[i][j] = C[i_base + i][j_base + j];
        }
    }
}

static void store_tile_C(
    double C[N][N],
    double tileC[TILE_N][TILE_N],
    int i_base, int j_base,
    int tile_rows, int tile_cols)
{
    for (int i = 0; i < tile_rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_N
        for (int j = 0; j < tile_cols; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_N
            C[i_base + i][j_base + j] = tileC[i][j];
        }
    }
}

static void load_tile_A(
    double A[N][M],
    double tileA_0[TILE_N][TILE_M],
    double tileA_1[TILE_N][TILE_M],
    int buf,
    int i_base, int k_base,
    int tile_rows, int tile_cols)
{
    if (buf == 0) {
        for (int i = 0; i < tile_rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_N
            for (int k = 0; k < tile_cols; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_M
                tileA_0[i][k] = A[i_base + i][k_base + k];
            }
        }
    } else {
        for (int i = 0; i < tile_rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_N
            for (int k = 0; k < tile_cols; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_M
                tileA_1[i][k] = A[i_base + i][k_base + k];
            }
        }
    }
}

static void load_tile_B(
    double B[N][M],
    double tileB_0[TILE_N][TILE_M],
    double tileB_1[TILE_N][TILE_M],
    int buf,
    int i_base, int k_base,
    int tile_rows, int tile_cols)
{
    if (buf == 0) {
        for (int i = 0; i < tile_rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_N
            for (int k = 0; k < tile_cols; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_M
                tileB_0[i][k] = B[i_base + i][k_base + k];
            }
        }
    } else {
        for (int i = 0; i < tile_rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_N
            for (int k = 0; k < tile_cols; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_M
                tileB_1[i][k] = B[i_base + i][k_base + k];
            }
        }
    }
}

static void compute_tile(
    double tileC[TILE_N][TILE_N],
    double tileA_i_0[TILE_N][TILE_M],
    double tileA_i_1[TILE_N][TILE_M],
    double tileA_j_0[TILE_N][TILE_M],
    double tileA_j_1[TILE_N][TILE_M],
    double tileB_i_0[TILE_N][TILE_M],
    double tileB_i_1[TILE_N][TILE_M],
    double tileB_j_0[TILE_N][TILE_M],
    double tileB_j_1[TILE_N][TILE_M],
    int buf,
    double alpha,
    int i_base, int j_base,
    int ti_rows, int tj_cols, int tk_cols)
{
    compute_i: for (int i = 0; i < ti_rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_N
        compute_j: for (int j = 0; j < tj_cols; j++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_N
            if ((i_base + i) >= (j_base + j)) {
                compute_k: for (int k = 0; k < tk_cols; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_M
#pragma HLS DEPENDENCE variable=tileC inter false
                    double Aik, Bik, Ajk, Bjk;
                    if (buf == 0) {
                        Aik = tileA_i_0[i][k];
                        Bik = tileB_i_0[i][k];
                        Ajk = tileA_j_0[j][k];
                        Bjk = tileB_j_0[j][k];
                    } else {
                        Aik = tileA_i_1[i][k];
                        Bik = tileB_i_1[i][k];
                        Ajk = tileA_j_1[j][k];
                        Bjk = tileB_j_1[j][k];
                    }
                    tileC[i][j] += Ajk * alpha * Bik + Bjk * alpha * Aik;
                }
            }
        }
    }
}

void kernel_syr2k(
          double alpha,
          double beta,
          double C[N + 0][N + 0],
          double A[N + 0][M + 0],
          double B[N + 0][M + 0])
{
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem0 \
    max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem1 \
    max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem2 \
    max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE s_axilite port=alpha bundle=control
#pragma HLS INTERFACE s_axilite port=beta  bundle=control
#pragma HLS INTERFACE s_axilite port=C     bundle=control
#pragma HLS INTERFACE s_axilite port=A     bundle=control
#pragma HLS INTERFACE s_axilite port=B     bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local tile buffers for C
    double tileC[TILE_N][TILE_N];

    // Double buffers for A_i, B_i, A_j, B_j (ping-pong)
    double tileA_i_0[TILE_N][TILE_M];
    double tileA_i_1[TILE_N][TILE_M];
    double tileA_j_0[TILE_N][TILE_M];
    double tileA_j_1[TILE_N][TILE_M];
    double tileB_i_0[TILE_N][TILE_M];
    double tileB_i_1[TILE_N][TILE_M];
    double tileB_j_0[TILE_N][TILE_M];
    double tileB_j_1[TILE_N][TILE_M];

#pragma HLS ARRAY_PARTITION variable=tileC     cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=tileA_i_0 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=tileA_i_1 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=tileA_j_0 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=tileA_j_1 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=tileB_i_0 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=tileB_i_1 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=tileB_j_0 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=tileB_j_1 cyclic factor=8 dim=2

    tile_i: for (int i_base = 0; i_base < N; i_base += TILE_N) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=N/TILE_N
        int i_end = (i_base + TILE_N < N) ? i_base + TILE_N : N;
        int ti_rows = i_end - i_base;

        tile_j: for (int j_base = 0; j_base <= i_base; j_base += TILE_N) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=N/TILE_N
            int j_end = (j_base + TILE_N < i_end) ? j_base + TILE_N : i_end;
            if (j_end > N) j_end = N;
            int tj_cols = j_end - j_base;

            // --- LOAD phase: load the C tile ---
            load_tile_C(C, tileC, i_base, j_base, ti_rows, tj_cols);

            // --- COMPUTE phase: beta scaling ---
            beta_scale_i: for (int i = 0; i < ti_rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_N
                beta_scale_j: for (int j = 0; j < tj_cols; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_N
#pragma HLS DEPENDENCE variable=tileC inter false
                    if ((i_base + i) >= (j_base + j)) {
                        tileC[i][j] *= beta;
                    }
                }
            }

            // --- Double-buffered tile_k loop ---
            int num_k_tiles = (M + TILE_M - 1) / TILE_M;

            if (num_k_tiles > 0) {
                // Pre-load first k-tile into buffer 0
                int k_base_0 = 0;
                int k_end_0 = (k_base_0 + TILE_M < M) ? k_base_0 + TILE_M : M;
                int tk_cols_0 = k_end_0 - k_base_0;

                load_tile_A(A, tileA_i_0, tileA_i_1, 0, i_base, k_base_0, ti_rows, tk_cols_0);
                load_tile_B(B, tileB_i_0, tileB_i_1, 0, i_base, k_base_0, ti_rows, tk_cols_0);
                load_tile_A(A, tileA_j_0, tileA_j_1, 0, j_base, k_base_0, tj_cols, tk_cols_0);
                load_tile_B(B, tileB_j_0, tileB_j_1, 0, j_base, k_base_0, tj_cols, tk_cols_0);

                tile_k: for (int k_base = 0; k_base < M; k_base += TILE_M) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=M/TILE_M
                    int k_end = (k_base + TILE_M < M) ? k_base + TILE_M : M;
                    int tk_cols = k_end - k_base;

                    // Current buffer index: 0 for even tiles, 1 for odd tiles
                    int cur_buf = ((k_base / TILE_M) % 2);

                    // Compute on current buffer (cur_buf)
                    compute_tile(
                        tileC,
                        tileA_i_0, tileA_i_1,
                        tileA_j_0, tileA_j_1,
                        tileB_i_0, tileB_i_1,
                        tileB_j_0, tileB_j_1,
                        cur_buf,
                        alpha,
                        i_base, j_base,
                        ti_rows, tj_cols, tk_cols);

                    // Pre-load next k-tile into the other buffer (1 - cur_buf)
                    int k_base_next = k_base + TILE_M;
                    if (k_base_next < M) {
                        int k_end_next = (k_base_next + TILE_M < M) ? k_base_next + TILE_M : M;
                        int tk_cols_next = k_end_next - k_base_next;
                        int next_buf = 1 - cur_buf;

                        load_tile_A(A, tileA_i_0, tileA_i_1, next_buf, i_base, k_base_next, ti_rows, tk_cols_next);
                        load_tile_B(B, tileB_i_0, tileB_i_1, next_buf, i_base, k_base_next, ti_rows, tk_cols_next);
                        load_tile_A(A, tileA_j_0, tileA_j_1, next_buf, j_base, k_base_next, tj_cols, tk_cols_next);
                        load_tile_B(B, tileB_j_0, tileB_j_1, next_buf, j_base, k_base_next, tj_cols, tk_cols_next);
                    }
                }
            }

            // --- STORE phase: write C tile back to global memory ---
            store_tile_C(C, tileC, i_base, j_base, ti_rows, tj_cols);
        }
    }
}