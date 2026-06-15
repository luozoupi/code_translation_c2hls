#include "jacobi-2d.h"
#include <string.h>

static void load(double lA[N][N], double lB[N][N],
                 double A[N][N], double B[N][N]) {
    const int n = N;
    load_A_rows:
    for (int i = 0; i < n; i++) {
        load_A_cols:
        for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            lA[i][j] = A[i][j];
        }
    }
    load_B_rows:
    for (int i = 0; i < n; i++) {
        load_B_cols:
        for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            lB[i][j] = B[i][j];
        }
    }
}

static void compute(double lA[N][N], double lB[N][N]) {
    const int n = N;
    const int tsteps = TSTEPS;

    const int TILE_ROWS = 16;

    for (int t = 0; t < tsteps; t++) {
        // A -> B stencil, processed in row tiles
        stencil_AB_tile:
        for (int ti = 1; ti < n - 1; ti += TILE_ROWS) {
#pragma HLS LOOP_TRIPCOUNT min=6 max=6
            int i_end = (ti + TILE_ROWS < n - 1) ? ti + TILE_ROWS : n - 1;
            stencil_AB_outer:
            for (int i = ti; i < i_end; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=16
                stencil_AB_inner:
                for (int j = 1; j < n - 1; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS DEPENDENCE variable=lA inter false
#pragma HLS DEPENDENCE variable=lB inter false
#pragma HLS LOOP_TRIPCOUNT min=88 max=88
                    lB[i][j] = 0.2 * (lA[i][j] + lA[i][j-1] + lA[i][j+1]
                                     + lA[i+1][j] + lA[i-1][j]);
                }
            }
        }

        // B -> A stencil, processed in row tiles
        stencil_BA_tile:
        for (int ti = 1; ti < n - 1; ti += TILE_ROWS) {
#pragma HLS LOOP_TRIPCOUNT min=6 max=6
            int i_end = (ti + TILE_ROWS < n - 1) ? ti + TILE_ROWS : n - 1;
            stencil_BA_outer:
            for (int i = ti; i < i_end; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=16
                stencil_BA_inner:
                for (int j = 1; j < n - 1; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS DEPENDENCE variable=lB inter false
#pragma HLS DEPENDENCE variable=lA inter false
#pragma HLS LOOP_TRIPCOUNT min=88 max=88
                    lA[i][j] = 0.2 * (lB[i][j] + lB[i][j-1] + lB[i][j+1]
                                     + lB[i+1][j] + lB[i-1][j]);
                }
            }
        }
    }
}

static void store(double lA[N][N], double lB[N][N],
                  double A[N][N], double B[N][N]) {
    const int n = N;
    store_A_rows:
    for (int i = 0; i < n; i++) {
        store_A_cols:
        for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            A[i][j] = lA[i][j];
        }
    }
    store_B_rows:
    for (int i = 0; i < n; i++) {
        store_B_cols:
        for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            B[i][j] = lB[i][j];
        }
    }
}

extern "C" {

void kernel_jacobi_2d(
                double A[N + 0][N + 0],
                double B[N + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local tile buffers for the entire working set
    double lA[N][N];
    double lB[N][N];

// Increase partition factor to 8 to support unroll factor=4 with stencil neighbors
// (each unrolled iteration accesses j-1, j, j+1 so we need enough banks)
#pragma HLS ARRAY_PARTITION variable=lA cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=lB cyclic factor=8 dim=2

    // Phase 1: Load global memory into local tile buffers
    load(lA, lB, A, B);

    // Phase 2: Compute stencil on local tile buffers
    compute(lA, lB);

    // Phase 3: Store local tile buffers back to global memory
    store(lA, lB, A, B);
}

} // extern "C"