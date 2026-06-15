#include "mvt.h"

// The kernel_mvt signature is declared in mvt.h and defined here.
// We preserve the algorithm exactly (serial FP reduction — no reassociation).
void kernel_mvt(
        double x1[ N + 0],
        double x2[ N + 0],
        double y_1[ N + 0],
        double y_2[ N + 0],
        double A[ N + 0][N + 0])
{
#pragma HLS INLINE off

    // Partition local-access arrays for parallel read inside pipelined inner loops
#pragma HLS ARRAY_PARTITION variable=x1  complete dim=1
#pragma HLS ARRAY_PARTITION variable=x2  complete dim=1
#pragma HLS ARRAY_PARTITION variable=y_1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=y_2 complete dim=1
    // Cyclic partition of A on dim=2 (column) so consecutive j-accesses hit
    // different banks; dim=1 cyclic helps the transpose loop (A[j][i]).
#pragma HLS ARRAY_PARTITION variable=A cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=A cyclic factor=8 dim=1

    const int n = N;

    int i, j;

    // Loop 1: x1[i] += A[i][j] * y_1[j]  — serial FP reduction over j
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            x1[i] = x1[i] + A[i][j] * y_1[j];
        }
    }

    // Loop 2: x2[i] += A[j][i] * y_2[j]  — serial FP reduction over j
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            x2[i] = x2[i] + A[j][i] * y_2[j];
        }
    }
}

// Top-level workload wrapper with AXI interfaces for Vitis HLS / Alveo U280
extern "C" {
void workload(
        double x1[ N + 0],
        double x2[ N + 0],
        double y_1[ N + 0],
        double y_2[ N + 0],
        double A[ N * N])   // flattened for m_axi port
{
#pragma HLS INTERFACE m_axi port=x1  offset=slave bundle=gmem0 depth=120
#pragma HLS INTERFACE m_axi port=x2  offset=slave bundle=gmem1 depth=120
#pragma HLS INTERFACE m_axi port=y_1 offset=slave bundle=gmem2 depth=120
#pragma HLS INTERFACE m_axi port=y_2 offset=slave bundle=gmem3 depth=120
#pragma HLS INTERFACE m_axi port=A   offset=slave bundle=gmem4 depth=14400

#pragma HLS INTERFACE s_axilite port=x1     bundle=control
#pragma HLS INTERFACE s_axilite port=x2     bundle=control
#pragma HLS INTERFACE s_axilite port=y_1    bundle=control
#pragma HLS INTERFACE s_axilite port=y_2    bundle=control
#pragma HLS INTERFACE s_axilite port=A      bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local buffers to stage data from global memory
    double lx1[N], lx2[N], ly1[N], ly2[N];
    double lA[N][N];

#pragma HLS ARRAY_PARTITION variable=lx1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=lx2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=ly1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=ly2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=lA cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=lA cyclic factor=8 dim=1

    // Load x1
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        lx1[i] = x1[i];
    }
    // Load x2
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        lx2[i] = x2[i];
    }
    // Load y_1
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        ly1[i] = y_1[i];
    }
    // Load y_2
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        ly2[i] = y_2[i];
    }
    // Load A (flattened)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            lA[i][j] = A[i * N + j];
        }
    }

    // Compute using the original kernel
    kernel_mvt(lx1, lx2, ly1, ly2, lA);

    // Store x1 back
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        x1[i] = lx1[i];
    }
    // Store x2 back
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        x2[i] = lx2[i];
    }
}
} // extern "C"