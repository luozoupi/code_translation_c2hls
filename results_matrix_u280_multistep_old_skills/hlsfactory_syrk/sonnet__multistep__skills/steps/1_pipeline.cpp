#include "syrk.h"

static const int TILE_K = 16;   // tile size along k dimension
static const int TILE_I = 16;   // tile size along i (row) dimension

// Load a tile of A: rows [i_base, i_base+TILE_I) x cols [k_base, k_base+TILE_K)
static void load_A_tile(double A[N][M],
                         double A_tile[TILE_I][TILE_K],
                         int i_base, int k_base,
                         int tile_rows, int tile_cols)
{
    load_A_i: for (int i = 0; i < tile_rows; i++) {
        load_A_k: for (int k = 0; k < tile_cols; k++) {
            #pragma HLS PIPELINE II=1
            A_tile[i][k] = A[i_base + i][k_base + k];
        }
    }
}

// Load a tile of C (lower triangular region): rows [i_base, i_base+TILE_I) x cols [0, i_base+TILE_I)
static void load_C_tile(double C[N][N],
                         double C_tile[TILE_I][N],
                         int i_base, int tile_rows)
{
    load_C_i: for (int i = 0; i < tile_rows; i++) {
        load_C_j: for (int j = 0; j <= i_base + i; j++) {
            #pragma HLS PIPELINE II=1
            C_tile[i][j] = C[i_base + i][j];
        }
    }
}

// Store a tile of C back: rows [i_base, i_base+TILE_I) x cols [0, i_base+TILE_I)
static void store_C_tile(double C[N][N],
                          double C_tile[TILE_I][N],
                          int i_base, int tile_rows)
{
    store_C_i: for (int i = 0; i < tile_rows; i++) {
        store_C_j: for (int j = 0; j <= i_base + i; j++) {
            #pragma HLS PIPELINE II=1
            C[i_base + i][j] = C_tile[i][j];
        }
    }
}

void kernel_syrk(
         double alpha,
         double beta,
         double C[N + 0][N + 0],
         double A[N + 0][M + 0])
{
    #pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem1
    #pragma HLS INTERFACE s_axilite port=alpha bundle=control
    #pragma HLS INTERFACE s_axilite port=beta bundle=control
    #pragma HLS INTERFACE s_axilite port=C bundle=control
    #pragma HLS INTERFACE s_axilite port=A bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local tile buffers
    double A_tile_i[TILE_I][TILE_K];   // tile of A for i-rows
    double A_tile_j[TILE_I][TILE_K];   // tile of A for j-rows (reuse across j tiles)
    double C_tile[TILE_I][N];          // working rows of C (lower triangular columns)

    #pragma HLS ARRAY_PARTITION variable=A_tile_i complete dim=2
    #pragma HLS ARRAY_PARTITION variable=A_tile_j complete dim=2
    #pragma HLS ARRAY_PARTITION variable=C_tile cyclic factor=8 dim=2
    // Partition A_tile_i and A_tile_j along dim=1 to allow parallel row access in compute loops
    #pragma HLS ARRAY_PARTITION variable=A_tile_i complete dim=1
    #pragma HLS ARRAY_PARTITION variable=A_tile_j complete dim=1

    // Process output rows in tiles of TILE_I
    tile_i: for (int i_base = 0; i_base < N; i_base += TILE_I) {
        #pragma HLS LOOP_TRIPCOUNT min=5 max=5
        int tile_rows_i = (i_base + TILE_I <= N) ? TILE_I : (N - i_base);

        // --- LOAD phase: load C tile (rows i_base..i_base+tile_rows_i-1, lower tri) ---
        load_C_i: for (int i = 0; i < tile_rows_i; i++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=16
            load_C_j: for (int j = 0; j <= i_base + i; j++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=1 max=80
                C_tile[i][j] = C[i_base + i][j];
            }
        }

        // --- COMPUTE phase: scale C by beta ---
        scale_i: for (int i = 0; i < tile_rows_i; i++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=16
            scale_j: for (int j = 0; j <= i_base + i; j++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=1 max=80
                #pragma HLS DEPENDENCE variable=C_tile inter false
                C_tile[i][j] *= beta;
            }
        }

        // --- COMPUTE phase: accumulate alpha * A[i,:] * A[j,:]^T over k-tiles ---
        // Tile over k dimension
        tile_k: for (int k_base = 0; k_base < M; k_base += TILE_K) {
            #pragma HLS LOOP_TRIPCOUNT min=4 max=4
            int tile_cols_k = (k_base + TILE_K <= M) ? TILE_K : (M - k_base);

            // Load A tile for i-rows: A[i_base..i_base+tile_rows_i-1][k_base..k_base+tile_cols_k-1]
            load_Ai_i: for (int i = 0; i < tile_rows_i; i++) {
                #pragma HLS LOOP_TRIPCOUNT min=1 max=16
                load_Ai_k: for (int k = 0; k < tile_cols_k; k++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=16
                    A_tile_i[i][k] = A[i_base + i][k_base + k];
                }
            }

            // Process j-row tiles from 0 up to i_base+tile_rows_i (lower triangular)
            tile_j: for (int j_base = 0; j_base <= i_base; j_base += TILE_I) {
                #pragma HLS LOOP_TRIPCOUNT min=1 max=5
                int j_end = (j_base + TILE_I <= i_base + tile_rows_i) ?
                             (j_base + TILE_I) : (i_base + tile_rows_i);
                int tile_rows_j = j_end - j_base;

                // Load A tile for j-rows: A[j_base..j_end-1][k_base..k_base+tile_cols_k-1]
                load_Aj_j: for (int jj = 0; jj < tile_rows_j; jj++) {
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=16
                    load_Aj_k: for (int k = 0; k < tile_cols_k; k++) {
                        #pragma HLS PIPELINE II=1
                        #pragma HLS LOOP_TRIPCOUNT min=1 max=16
                        A_tile_j[jj][k] = A[j_base + jj][k_base + k];
                    }
                }

                // Compute partial dot products: C_tile[i][j] += alpha * A[i,k] * A[j,k]
                // Only update lower triangular: j <= i (global indices)
                compute_i: for (int i = 0; i < tile_rows_i; i++) {
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=16
                    int global_i = i_base + i;
                    compute_j: for (int jj = 0; jj < tile_rows_j; jj++) {
                        #pragma HLS PIPELINE II=1
                        #pragma HLS LOOP_TRIPCOUNT min=1 max=16
                        #pragma HLS DEPENDENCE variable=C_tile inter false
                        int global_j = j_base + jj;
                        if (global_j <= global_i) {
                            double acc = 0.0;
                            compute_k: for (int k = 0; k < TILE_K; k++) {
                                #pragma HLS UNROLL
                                acc += A_tile_i[i][k] * A_tile_j[jj][k];
                            }
                            C_tile[i][global_j] += alpha * acc;
                        }
                    }
                }
            }
        }

        // --- STORE phase: write C tile back to global memory ---
        store_C_i: for (int i = 0; i < tile_rows_i; i++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=16
            store_C_j: for (int j = 0; j <= i_base + i; j++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=1 max=80
                C[i_base + i][j] = C_tile[i][j];
            }
        }
    }
}