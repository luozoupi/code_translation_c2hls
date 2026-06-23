#include "trmm.h"
#include <cstring>

void kernel_trmm( 
		 double alpha,
		 double A[ M + 0][M + 0],
		 double B[ M + 0][N + 0])
{
#pragma HLS INLINE off

    const int m = M;
    const int n = N;

    // Local buffers staged from global memory
    static double A_local[M][M];
    static double B_local[M][N];
    // Partition local arrays so the inner reduction can read multiple
    // operands per cycle and to relieve memory-port conflicts in the pipeline.
#pragma HLS ARRAY_PARTITION variable=A_local cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=B_local cyclic factor=4 dim=1
    // Partition along the j/n dimension to feed the unrolled compute_j loop.
#pragma HLS ARRAY_PARTITION variable=B_local cyclic factor=4 dim=2

    int i, j, k;

    // ---- LOAD phase: stage A and B into local memory ----
    load_A_rows:
    for (i = 0; i < m; i++) {
        load_A_cols:
        for (j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
            A_local[i][j] = A[i][j];
        }
    }

    load_B_rows:
    for (i = 0; i < m; i++) {
        load_B_cols:
        for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            B_local[i][j] = B[i][j];
        }
    }

    // ---- COMPUTE phase: operate on local buffers ----
    compute_i:
    for (i = 0; i < m; i++) {
        compute_j:
        for (j = 0; j < n; j++) {
#pragma HLS UNROLL factor=4
            double acc = B_local[i][j];
            compute_k:
            for (k = i + 1; k < m; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=0 max=M
                acc += A_local[k][i] * B_local[k][j];
            }
            B_local[i][j] = alpha * acc;
        }
    }

    // ---- STORE phase: write results back to global memory ----
    store_B_rows:
    for (i = 0; i < m; i++) {
        store_B_cols:
        for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            B[i][j] = B_local[i][j];
        }
    }
}

extern "C" {
void workload(
		 double alpha,
		 double A[ M + 0][M + 0],
		 double B[ M + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=alpha bundle=control
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    kernel_trmm(alpha, A, B);
}
}