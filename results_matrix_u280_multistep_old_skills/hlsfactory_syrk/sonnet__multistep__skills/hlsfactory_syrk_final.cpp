#include "syrk.h"

static const int TILE_K = 16;   // tile size along k dimension
static const int TILE_I = 16;   // tile size along i (row) dimension

void kernel_syrk(
         double alpha,
         double beta,
         double C[N + 0][N + 0],
         double A[N + 0][M + 0])
{
    #pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem  max_read_burst_length=256 max_write_burst_length=256
    #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem1 max_read_burst_length=256 max_write_burst_length=256
    #pragma HLS INTERFACE s_axilite port=alpha bundle=control
    #pragma HLS INTERFACE s_axilite port=beta bundle=control
    #pragma HLS INTERFACE s_axilite port=C bundle=control
    #pragma HLS INTERFACE s_axilite port=A bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // Double-buffered A tiles for i-rows (ping-pong)
    double A_tile_i_0[TILE_I][TILE_K];
    double A_tile_i_1[TILE_I][TILE_K];

    // Double-buffered A tiles for j-rows (ping-pong)
    double A_tile_j_0[TILE_I][TILE_K];
    double A_tile_j_1[TILE_I][TILE_K];

    // Partition A_tile_i buffers: complete along k dim, cyclic factor=4 along row dim
    #pragma HLS ARRAY_PARTITION variable=A_tile_i_0 complete dim=2
    #pragma HLS ARRAY_PARTITION variable=A_tile_i_1 complete dim=2
    #pragma HLS ARRAY_PARTITION variable=A_tile_i_0 cyclic factor=4 dim=1
    #pragma HLS ARRAY_PARTITION variable=A_tile_i_1 cyclic factor=4 dim=1

    // Partition A_tile_j buffers: complete along k dim, cyclic factor=4 along row dim
    #pragma HLS ARRAY_PARTITION variable=A_tile_j_0 complete dim=2
    #pragma HLS ARRAY_PARTITION variable=A_tile_j_1 complete dim=2
    #pragma HLS ARRAY_PARTITION variable=A_tile_j_0 cyclic factor=4 dim=1
    #pragma HLS ARRAY_PARTITION variable=A_tile_j_1 cyclic factor=4 dim=1

    // C_tile buffer (single, no double-buffer needed here)
    double C_tile[TILE_I][N];
    #pragma HLS ARRAY_PARTITION variable=C_tile cyclic factor=8 dim=2
    #pragma HLS ARRAY_PARTITION variable=C_tile cyclic factor=4 dim=1

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

        // --- Double-buffered COMPUTE phase over k-tiles ---
        // Preload the first k-tile into buffer set 0 before entering the main loop
        {
            int k_base_pre = 0;
            int tile_cols_pre = (k_base_pre + TILE_K <= M) ? TILE_K : (M - k_base_pre);

            // Preload A_tile_i into buffer 0
            preload_Ai_i: for (int i = 0; i < tile_rows_i; i++) {
                #pragma HLS LOOP_TRIPCOUNT min=1 max=16
                preload_Ai_k: for (int k = 0; k < tile_cols_pre; k++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=16
                    A_tile_i_0[i][k] = A[i_base + i][k_base_pre + k];
                }
            }
        }

        // Main k-tile loop with double buffering
        tile_k: for (int k_base = 0; k_base < M; k_base += TILE_K) {
            #pragma HLS LOOP_TRIPCOUNT min=4 max=4
            int tile_cols_k = (k_base + TILE_K <= M) ? TILE_K : (M - k_base);
            int cur_buf = (k_base / TILE_K) % 2;  // which buffer holds current data
            int next_buf = 1 - cur_buf;             // which buffer to preload next data into

            // Prefetch next A_tile_i (k+TILE_K) into next_buf
            int k_next = k_base + TILE_K;
            if (k_next < M) {
                int tile_cols_next = (k_next + TILE_K <= M) ? TILE_K : (M - k_next);
                prefetch_Ai_i: for (int i = 0; i < tile_rows_i; i++) {
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=16
                    prefetch_Ai_k: for (int k = 0; k < tile_cols_next; k++) {
                        #pragma HLS PIPELINE II=1
                        #pragma HLS LOOP_TRIPCOUNT min=1 max=16
                        if (next_buf == 0)
                            A_tile_i_0[i][k] = A[i_base + i][k_next + k];
                        else
                            A_tile_i_1[i][k] = A[i_base + i][k_next + k];
                    }
                }
            }

            // Process j-row tiles (lower triangular), with double buffering on A_tile_j
            // Preload first j-tile of A_tile_j into buffer 0
            {
                int j_base_pre = 0;
                int j_end_pre = (j_base_pre + TILE_I <= i_base + tile_rows_i) ?
                                 (j_base_pre + TILE_I) : (i_base + tile_rows_i);
                int tile_rows_j_pre = j_end_pre - j_base_pre;

                preload_Aj_j: for (int jj = 0; jj < tile_rows_j_pre; jj++) {
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=16
                    preload_Aj_k: for (int k = 0; k < tile_cols_k; k++) {
                        #pragma HLS PIPELINE II=1
                        #pragma HLS LOOP_TRIPCOUNT min=1 max=16
                        A_tile_j_0[jj][k] = A[j_base_pre + jj][k_base + k];
                    }
                }
            }

            tile_j: for (int j_base = 0; j_base <= i_base; j_base += TILE_I) {
                #pragma HLS LOOP_TRIPCOUNT min=1 max=5
                int j_end = (j_base + TILE_I <= i_base + tile_rows_i) ?
                             (j_base + TILE_I) : (i_base + tile_rows_i);
                int tile_rows_j = j_end - j_base;
                int cur_jbuf = (j_base / TILE_I) % 2;
                int next_jbuf = 1 - cur_jbuf;

                // Prefetch next j-tile of A_tile_j into next_jbuf
                int j_next = j_base + TILE_I;
                if (j_next <= i_base) {
                    int j_end_next = (j_next + TILE_I <= i_base + tile_rows_i) ?
                                     (j_next + TILE_I) : (i_base + tile_rows_i);
                    int tile_rows_j_next = j_end_next - j_next;
                    prefetch_Aj_j: for (int jj = 0; jj < tile_rows_j_next; jj++) {
                        #pragma HLS LOOP_TRIPCOUNT min=1 max=16
                        prefetch_Aj_k: for (int k = 0; k < tile_cols_k; k++) {
                            #pragma HLS PIPELINE II=1
                            #pragma HLS LOOP_TRIPCOUNT min=1 max=16
                            if (next_jbuf == 0)
                                A_tile_j_0[jj][k] = A[j_next + jj][k_base + k];
                            else
                                A_tile_j_1[jj][k] = A[j_next + jj][k_base + k];
                        }
                    }
                }

                // Compute partial dot products using current buffers
                compute_i: for (int i = 0; i < tile_rows_i; i++) {
                    #pragma HLS UNROLL factor=4
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
                                double ai = (cur_buf == 0) ? A_tile_i_0[i][k] : A_tile_i_1[i][k];
                                double aj = (cur_jbuf == 0) ? A_tile_j_0[jj][k] : A_tile_j_1[jj][k];
                                acc += ai * aj;
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