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
    double tileA[TILE_N][TILE_M],
    int i_base, int k_base,
    int tile_rows, int tile_cols)
{
    for (int i = 0; i < tile_rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_N
        for (int k = 0; k < tile_cols; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_M
            tileA[i][k] = A[i_base + i][k_base + k];
        }
    }
}

static void load_tile_B(
    double B[N][M],
    double tileB[TILE_N][TILE_M],
    int i_base, int k_base,
    int tile_rows, int tile_cols)
{
    for (int i = 0; i < tile_rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_N
        for (int k = 0; k < tile_cols; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_M
            tileB[i][k] = B[i_base + i][k_base + k];
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
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=alpha bundle=control
#pragma HLS INTERFACE s_axilite port=beta  bundle=control
#pragma HLS INTERFACE s_axilite port=C     bundle=control
#pragma HLS INTERFACE s_axilite port=A     bundle=control
#pragma HLS INTERFACE s_axilite port=B     bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local tile buffers
    double tileC[TILE_N][TILE_N];
    double tileA_i[TILE_N][TILE_M];
    double tileA_j[TILE_N][TILE_M];
    double tileB_i[TILE_N][TILE_M];
    double tileB_j[TILE_N][TILE_M];

    // Partition dim=2 (k-dimension) completely for full parallel read in pipeline
#pragma HLS ARRAY_PARTITION variable=tileC   cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=tileA_i complete dim=2
#pragma HLS ARRAY_PARTITION variable=tileA_j complete dim=2
#pragma HLS ARRAY_PARTITION variable=tileB_i complete dim=2
#pragma HLS ARRAY_PARTITION variable=tileB_j complete dim=2

    // Tile over output rows (i) and columns (j)
    tile_i: for (int i_base = 0; i_base < N; i_base += TILE_N) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=N/TILE_N
        int i_end = (i_base + TILE_N < N) ? i_base + TILE_N : N;
        int ti_rows = i_end - i_base;

        tile_j: for (int j_base = 0; j_base <= i_base; j_base += TILE_N) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=N/TILE_N
            // Only process lower triangle tiles
            int j_end = (j_base + TILE_N < i_end) ? j_base + TILE_N : i_end;
            // Clamp j_end to N
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
                    // Only scale lower-triangle elements
                    if ((i_base + i) >= (j_base + j)) {
                        tileC[i][j] *= beta;
                    }
                }
            }

            // --- COMPUTE phase: accumulate over k tiles ---
            tile_k: for (int k_base = 0; k_base < M; k_base += TILE_M) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=M/TILE_M
                int k_end = (k_base + TILE_M < M) ? k_base + TILE_M : M;
                int tk_cols = k_end - k_base;

                // Load A[i_base..i_end, k_base..k_end]
                load_tile_A(A, tileA_i, i_base, k_base, ti_rows, tk_cols);
                // Load B[i_base..i_end, k_base..k_end]
                load_tile_B(B, tileB_i, i_base, k_base, ti_rows, tk_cols);
                // Load A[j_base..j_end, k_base..k_end]
                load_tile_A(A, tileA_j, j_base, k_base, tj_cols, tk_cols);
                // Load B[j_base..j_end, k_base..k_end]
                load_tile_B(B, tileB_j, j_base, k_base, tj_cols, tk_cols);

                // Compute using local tiles
                compute_i: for (int i = 0; i < ti_rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_N
                    compute_j: for (int j = 0; j < tj_cols; j++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_N
                        // Only process lower-triangle
                        if ((i_base + i) >= (j_base + j)) {
                            compute_k: for (int k = 0; k < tk_cols; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_M
#pragma HLS DEPENDENCE variable=tileC inter false
                                double Aik = tileA_i[i][k];
                                double Bik = tileB_i[i][k];
                                double Ajk = tileA_j[j][k];
                                double Bjk = tileB_j[j][k];
                                tileC[i][j] += Ajk * alpha * Bik + Bjk * alpha * Aik;
                            }
                        }
                    }
                }
            }

            // --- STORE phase: write C tile back to global memory ---
            store_tile_C(C, tileC, i_base, j_base, ti_rows, tj_cols);
        }
    }
}