#include "symm.h"

extern "C" {

void kernel_symm(
        double alpha,
        double beta,
        double C[M][N],
        double A[M][M],
        double B[M][N])
{
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=alpha bundle=control
#pragma HLS INTERFACE s_axilite port=beta bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local row buffers to reduce repeated AXI accesses
    double A_row[M];
    double B_i_row[N];
    double C_i_row[N];
    double temp2_arr[N];

#pragma HLS ARRAY_PARTITION variable=A_row complete dim=1
#pragma HLS ARRAY_PARTITION variable=B_i_row complete dim=1
#pragma HLS ARRAY_PARTITION variable=C_i_row complete dim=1
#pragma HLS ARRAY_PARTITION variable=temp2_arr complete dim=1

    const int m = M;
    const int n = N;

    for (int i = 0; i < m; i++) {

        // Load A[i][*] into local buffer
        load_A: for (int k = 0; k < m; k++) {
#pragma HLS PIPELINE II=1
            A_row[k] = A[i][k];
        }

        // Load B[i][*] into local buffer
        load_B_i: for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            B_i_row[j] = B[i][j];
        }

        // Load C[i][*] into local buffer
        load_C_i: for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            C_i_row[j] = C[i][j];
        }

        // Initialize temp2 accumulators
        init_temp2: for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            temp2_arr[j] = 0.0;
        }

        // Inner k loop: for each k < i, update C[k][j] and accumulate temp2
        // Preserve serial FP reduction order (j varies fastest for a fixed k)
        for (int k = 0; k < i; k++) {
            double A_ik = A_row[k];
            double alpha_A_ik = alpha * A_ik;

            // Load B[k][*] row from global memory
            double B_k_row[N];
#pragma HLS ARRAY_PARTITION variable=B_k_row complete dim=1
            load_Bk: for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
                B_k_row[j] = B[k][j];
            }

            // Load C[k][*] row, update, store back
            double C_k_row[N];
#pragma HLS ARRAY_PARTITION variable=C_k_row complete dim=1
            load_Ck: for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
                C_k_row[j] = C[k][j];
            }

            // Update C[k][j] and temp2[j] — j iterations independent
            update_j: for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
                C_k_row[j] += alpha * B_i_row[j] * A_ik;
                temp2_arr[j] += B_k_row[j] * A_ik;
            }

            store_Ck: for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
                C[k][j] = C_k_row[j];
            }
        }

        // Final update for C[i][j]
        final_update: for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            C_i_row[j] = beta * C_i_row[j]
                         + alpha * B_i_row[j] * A_row[i]
                         + alpha * temp2_arr[j];
        }

        // Store C[i][*] back
        store_Ci: for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            C[i][j] = C_i_row[j];
        }
    }
}

} // extern "C"