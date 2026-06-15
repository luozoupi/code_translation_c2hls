#include "symm.h"

#define TILE_N 16

extern "C" {

void kernel_symm(
        double alpha,
        double beta,
        double C[M + 0][N + 0],
        double A[M + 0][M + 0],
        double B[M + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=alpha bundle=control
#pragma HLS INTERFACE s_axilite port=beta bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int m = M;
    const int n = N;

    // Local tile buffers - tile over columns (j-dimension) in chunks of TILE_N
    double A_row[M];           // A[i][0..i]  - full row of symmetric A

    // Double-buffered B_row_i: used to overlap load of next i with compute of current i
    double B_row_i_1[TILE_N];
    double B_row_i_2[TILE_N];

    // Double-buffered C_ij_orig: original C[i][jt..jt+TILE_N] before update
    double C_ij_orig_1[TILE_N];
    double C_ij_orig_2[TILE_N];

    double temp2_tile[TILE_N]; // accumulator for sum_k B[k][j]*A[i][k]

    // Double-buffered B_kj and C_kj for the inner k-loop
    double B_kj_1[TILE_N];
    double B_kj_2[TILE_N];
    double C_kj_1[TILE_N];
    double C_kj_2[TILE_N];

#pragma HLS ARRAY_PARTITION variable=A_row      cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=B_row_i_1  complete dim=1
#pragma HLS ARRAY_PARTITION variable=B_row_i_2  complete dim=1
#pragma HLS ARRAY_PARTITION variable=C_ij_orig_1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=C_ij_orig_2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=temp2_tile  complete dim=1
#pragma HLS ARRAY_PARTITION variable=B_kj_1      complete dim=1
#pragma HLS ARRAY_PARTITION variable=B_kj_2      complete dim=1
#pragma HLS ARRAY_PARTITION variable=C_kj_1      complete dim=1
#pragma HLS ARRAY_PARTITION variable=C_kj_2      complete dim=1

    // Tile loop over j (columns)
    for (int jt = 0; jt < n; jt += TILE_N) {
#pragma HLS LOOP_TRIPCOUNT min=5 max=5
        int j_len = ((jt + TILE_N) <= n) ? TILE_N : (n - jt);

        // Process each row i sequentially
        for (int i = 0; i < m; i++) {
#pragma HLS LOOP_TRIPCOUNT min=60 max=60

            // Select which buffer pair to use for B_row_i and C_ij_orig
            // based on i % 2
            int buf_sel = i % 2;

            // ==================================================
            // LOAD PHASE: load into the "current" buffer set
            // ==================================================

            // Load A[i][0..i] into local buffer
            load_A_row: for (int k = 0; k <= i; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=60
#pragma HLS DEPENDENCE variable=A_row inter false
                A_row[k] = A[i][k];
            }
            double A_ii = A_row[i];

            // Load B[i][jt..jt+j_len] into current tile buffer
            if (buf_sel == 0) {
                load_B_row_i_0: for (int j = 0; j < j_len; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS DEPENDENCE variable=B_row_i_1 inter false
                    B_row_i_1[j] = B[i][jt + j];
                }
            } else {
                load_B_row_i_1: for (int j = 0; j < j_len; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS DEPENDENCE variable=B_row_i_2 inter false
                    B_row_i_2[j] = B[i][jt + j];
                }
            }

            // Load C[i][jt..jt+j_len] into current tile buffer (original values)
            if (buf_sel == 0) {
                load_C_row_i_0: for (int j = 0; j < j_len; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS DEPENDENCE variable=C_ij_orig_1 inter false
                    C_ij_orig_1[j] = C[i][jt + j];
                }
            } else {
                load_C_row_i_1: for (int j = 0; j < j_len; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS DEPENDENCE variable=C_ij_orig_2 inter false
                    C_ij_orig_2[j] = C[i][jt + j];
                }
            }

            // Initialize temp2 accumulator tile to zero
            init_temp2: for (int j = 0; j < j_len; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS DEPENDENCE variable=temp2_tile inter false
                temp2_tile[j] = 0.0;
            }

            // ==================================================
            // COMPUTE PHASE: loop over k < i
            // Double-buffer B_kj and C_kj to overlap loads
            // ==================================================

            if (i > 0) {
                // Pre-load first k=0 into buffer set 0
                preload_Bk: for (int j = 0; j < j_len; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS DEPENDENCE variable=B_kj_1 inter false
                    B_kj_1[j] = B[0][jt + j];
                }
                preload_Ck: for (int j = 0; j < j_len; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS DEPENDENCE variable=C_kj_1 inter false
                    C_kj_1[j] = C[0][jt + j];
                }
            }

            compute_k: for (int k = 0; k < i; k++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=59
                double Aik = A_row[k];

                // Determine current and next buffer sets
                int k_sel = k % 2;      // current buffer set
                int k_next = k + 1;     // next k index

                // Compute using current buffer set (already loaded)
                if (k_sel == 0) {
                    // Compute from B_kj_1, C_kj_1
                    // Pre-load next k into B_kj_2, C_kj_2 (if k_next < i)
                    if (k_next < i) {
                        load_Bk_next_0: for (int j = 0; j < j_len; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS DEPENDENCE variable=B_kj_2 inter false
                            B_kj_2[j] = B[k_next][jt + j];
                        }
                        load_Ck_next_0: for (int j = 0; j < j_len; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS DEPENDENCE variable=C_kj_2 inter false
                            C_kj_2[j] = C[k_next][jt + j];
                        }
                    }

                    // Compute using B_row_i from current buf_sel
                    if (buf_sel == 0) {
                        compute_kj_00: for (int j = 0; j < j_len; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS DEPENDENCE variable=C_kj_1      inter false
#pragma HLS DEPENDENCE variable=temp2_tile  inter false
#pragma HLS DEPENDENCE variable=B_row_i_1   inter false
#pragma HLS DEPENDENCE variable=B_kj_1      inter false
                            C_kj_1[j]      += alpha * B_row_i_1[j] * Aik;
                            temp2_tile[j]  += B_kj_1[j] * Aik;
                        }
                    } else {
                        compute_kj_01: for (int j = 0; j < j_len; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS DEPENDENCE variable=C_kj_1      inter false
#pragma HLS DEPENDENCE variable=temp2_tile  inter false
#pragma HLS DEPENDENCE variable=B_row_i_2   inter false
#pragma HLS DEPENDENCE variable=B_kj_1      inter false
                            C_kj_1[j]      += alpha * B_row_i_2[j] * Aik;
                            temp2_tile[j]  += B_kj_1[j] * Aik;
                        }
                    }

                    // Store C[k] back
                    store_Ck_0: for (int j = 0; j < j_len; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS DEPENDENCE variable=C_kj_1 inter false
                        C[k][jt + j] = C_kj_1[j];
                    }

                } else {
                    // k_sel == 1: compute from B_kj_2, C_kj_2
                    // Pre-load next k into B_kj_1, C_kj_1 (if k_next < i)
                    if (k_next < i) {
                        load_Bk_next_1: for (int j = 0; j < j_len; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS DEPENDENCE variable=B_kj_1 inter false
                            B_kj_1[j] = B[k_next][jt + j];
                        }
                        load_Ck_next_1: for (int j = 0; j < j_len; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS DEPENDENCE variable=C_kj_1 inter false
                            C_kj_1[j] = C[k_next][jt + j];
                        }
                    }

                    // Compute using B_row_i from current buf_sel
                    if (buf_sel == 0) {
                        compute_kj_10: for (int j = 0; j < j_len; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS DEPENDENCE variable=C_kj_2      inter false
#pragma HLS DEPENDENCE variable=temp2_tile  inter false
#pragma HLS DEPENDENCE variable=B_row_i_1   inter false
#pragma HLS DEPENDENCE variable=B_kj_2      inter false
                            C_kj_2[j]      += alpha * B_row_i_1[j] * Aik;
                            temp2_tile[j]  += B_kj_2[j] * Aik;
                        }
                    } else {
                        compute_kj_11: for (int j = 0; j < j_len; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS DEPENDENCE variable=C_kj_2      inter false
#pragma HLS DEPENDENCE variable=temp2_tile  inter false
#pragma HLS DEPENDENCE variable=B_row_i_2   inter false
#pragma HLS DEPENDENCE variable=B_kj_2      inter false
                            C_kj_2[j]      += alpha * B_row_i_2[j] * Aik;
                            temp2_tile[j]  += B_kj_2[j] * Aik;
                        }
                    }

                    // Store C[k] back
                    store_Ck_1: for (int j = 0; j < j_len; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS DEPENDENCE variable=C_kj_2 inter false
                        C[k][jt + j] = C_kj_2[j];
                    }
                }
            }

            // ==================================================
            // COMPUTE + STORE PHASE: finalize C[i][j]
            // C[i][j] = beta*C_orig[i][j] + alpha*B[i][j]*A[i][i] + alpha*temp2[j]
            // ==================================================
            if (buf_sel == 0) {
                store_C_row_i_0: for (int j = 0; j < j_len; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS DEPENDENCE variable=C_ij_orig_1 inter false
#pragma HLS DEPENDENCE variable=B_row_i_1   inter false
#pragma HLS DEPENDENCE variable=temp2_tile  inter false
                    C[i][jt + j] = beta * C_ij_orig_1[j]
                                  + alpha * B_row_i_1[j] * A_ii
                                  + alpha * temp2_tile[j];
                }
            } else {
                store_C_row_i_1: for (int j = 0; j < j_len; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS DEPENDENCE variable=C_ij_orig_2 inter false
#pragma HLS DEPENDENCE variable=B_row_i_2   inter false
#pragma HLS DEPENDENCE variable=temp2_tile  inter false
                    C[i][jt + j] = beta * C_ij_orig_2[j]
                                  + alpha * B_row_i_2[j] * A_ii
                                  + alpha * temp2_tile[j];
                }
            }
        }
    }
}

} // extern "C"