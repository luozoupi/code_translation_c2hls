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
    double B_row_i[TILE_N];    // B[i][jt..jt+TILE_N]  - tile of row i of B
    double B_kj[TILE_N];       // B[k][jt..jt+TILE_N]  - tile of row k of B
    double C_kj[TILE_N];       // C[k][jt..jt+TILE_N]  - tile of row k of C (for update)
    double C_ij_orig[TILE_N];  // original C[i][jt..jt+TILE_N] before update
    double temp2_tile[TILE_N]; // accumulator for sum_k B[k][j]*A[i][k]

#pragma HLS ARRAY_PARTITION variable=A_row      cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=B_row_i    complete dim=1
#pragma HLS ARRAY_PARTITION variable=B_kj       complete dim=1
#pragma HLS ARRAY_PARTITION variable=C_kj       complete dim=1
#pragma HLS ARRAY_PARTITION variable=C_ij_orig  complete dim=1
#pragma HLS ARRAY_PARTITION variable=temp2_tile complete dim=1

    // Tile loop over j (columns)
    for (int jt = 0; jt < n; jt += TILE_N) {
#pragma HLS LOOP_TRIPCOUNT min=5 max=5
        int j_len = ((jt + TILE_N) <= n) ? TILE_N : (n - jt);

        // Process each row i sequentially
        for (int i = 0; i < m; i++) {
#pragma HLS LOOP_TRIPCOUNT min=60 max=60

            // ==================================================
            // LOAD PHASE
            // ==================================================

            // Load A[i][0..i] into local buffer
            load_A_row: for (int k = 0; k <= i; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=60
#pragma HLS DEPENDENCE variable=A_row inter false
                A_row[k] = A[i][k];
            }
            double A_ii = A_row[i];

            // Load B[i][jt..jt+j_len] into local tile
            load_B_row_i: for (int j = 0; j < j_len; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS DEPENDENCE variable=B_row_i inter false
                B_row_i[j] = B[i][jt + j];
            }

            // Load C[i][jt..jt+j_len] into local tile (original values)
            load_C_row_i: for (int j = 0; j < j_len; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS DEPENDENCE variable=C_ij_orig inter false
                C_ij_orig[j] = C[i][jt + j];
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
            // For each k: update C[k][j] and accumulate temp2
            // ==================================================
            compute_k: for (int k = 0; k < i; k++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=59
                double Aik = A_row[k];

                // Load B[k][jt..jt+j_len]
                load_Bk: for (int j = 0; j < j_len; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS DEPENDENCE variable=B_kj inter false
                    B_kj[j] = B[k][jt + j];
                }

                // Load C[k][jt..jt+j_len]
                load_Ck: for (int j = 0; j < j_len; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS DEPENDENCE variable=C_kj inter false
                    C_kj[j] = C[k][jt + j];
                }

                // Compute: C[k][j] += alpha * B[i][j] * A[i][k]
                //          temp2[j] += B[k][j] * A[i][k]
                compute_kj: for (int j = 0; j < j_len; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS DEPENDENCE variable=C_kj       inter false
#pragma HLS DEPENDENCE variable=temp2_tile inter false
#pragma HLS DEPENDENCE variable=B_row_i    inter false
#pragma HLS DEPENDENCE variable=B_kj       inter false
                    C_kj[j]      += alpha * B_row_i[j] * Aik;
                    temp2_tile[j] += B_kj[j] * Aik;
                }

                // Store C[k][jt..jt+j_len] back to global memory
                store_Ck: for (int j = 0; j < j_len; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS DEPENDENCE variable=C_kj inter false
                    C[k][jt + j] = C_kj[j];
                }
            }

            // ==================================================
            // COMPUTE + STORE PHASE: finalize C[i][j]
            // C[i][j] = beta*C_orig[i][j] + alpha*B[i][j]*A[i][i] + alpha*temp2[j]
            // ==================================================
            store_C_row_i: for (int j = 0; j < j_len; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS DEPENDENCE variable=C_ij_orig  inter false
#pragma HLS DEPENDENCE variable=B_row_i    inter false
#pragma HLS DEPENDENCE variable=temp2_tile inter false
                C[i][jt + j] = beta * C_ij_orig[j]
                              + alpha * B_row_i[j] * A_ii
                              + alpha * temp2_tile[j];
            }
        }
    }
}

} // extern "C"