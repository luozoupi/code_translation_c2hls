#include "jacobi-2d.h"
#include <string.h>

static void load(int buf_sel,
                 double lA0[N][N], double lB0[N][N],
                 double lA1[N][N], double lB1[N][N],
                 double A[N][N], double B[N][N]) {
    const int n = N;
    if (buf_sel == 0) {
        load_A_rows_0:
        for (int i = 0; i < n; i++) {
            load_A_cols_0:
            for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
                lA0[i][j] = A[i][j];
            }
        }
        load_B_rows_0:
        for (int i = 0; i < n; i++) {
            load_B_cols_0:
            for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
                lB0[i][j] = B[i][j];
            }
        }
    } else {
        load_A_rows_1:
        for (int i = 0; i < n; i++) {
            load_A_cols_1:
            for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
                lA1[i][j] = A[i][j];
            }
        }
        load_B_rows_1:
        for (int i = 0; i < n; i++) {
            load_B_cols_1:
            for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
                lB1[i][j] = B[i][j];
            }
        }
    }
}

static void compute(int buf_sel,
                    double lA0[N][N], double lB0[N][N],
                    double lA1[N][N], double lB1[N][N]) {
    const int n = N;
    const int TILE_ROWS = 16;

    double (*lA)[N] = (buf_sel == 0) ? lA0 : lA1;
    double (*lB)[N] = (buf_sel == 0) ? lB0 : lB1;

    for (int t = 0; t < TSTEPS; t++) {
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
#pragma HLS DEPENDENCE variable=lA0 inter false
#pragma HLS DEPENDENCE variable=lB0 inter false
#pragma HLS DEPENDENCE variable=lA1 inter false
#pragma HLS DEPENDENCE variable=lB1 inter false
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
#pragma HLS DEPENDENCE variable=lB0 inter false
#pragma HLS DEPENDENCE variable=lA0 inter false
#pragma HLS DEPENDENCE variable=lB1 inter false
#pragma HLS DEPENDENCE variable=lA1 inter false
#pragma HLS LOOP_TRIPCOUNT min=88 max=88
                    lA[i][j] = 0.2 * (lB[i][j] + lB[i][j-1] + lB[i][j+1]
                                     + lB[i+1][j] + lB[i-1][j]);
                }
            }
        }
    }
}

static void store(int buf_sel,
                  double lA0[N][N], double lB0[N][N],
                  double lA1[N][N], double lB1[N][N],
                  double A[N][N], double B[N][N]) {
    const int n = N;
    if (buf_sel == 0) {
        store_A_rows_0:
        for (int i = 0; i < n; i++) {
            store_A_cols_0:
            for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
                A[i][j] = lA0[i][j];
            }
        }
        store_B_rows_0:
        for (int i = 0; i < n; i++) {
            store_B_cols_0:
            for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
                B[i][j] = lB0[i][j];
            }
        }
    } else {
        store_A_rows_1:
        for (int i = 0; i < n; i++) {
            store_A_cols_1:
            for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
                A[i][j] = lA1[i][j];
            }
        }
        store_B_rows_1:
        for (int i = 0; i < n; i++) {
            store_B_cols_1:
            for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
                B[i][j] = lB1[i][j];
            }
        }
    }
}

extern "C" {

void kernel_jacobi_2d(
                double A[N + 0][N + 0],
                double B[N + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Double (ping-pong) local tile buffers
    double lA0[N][N];
    double lB0[N][N];
    double lA1[N][N];
    double lB1[N][N];

#pragma HLS ARRAY_PARTITION variable=lA0 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=lB0 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=lA1 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=lB1 cyclic factor=8 dim=2

    // Phase 1: Load global memory into ping buffer (buf_sel=0)
    load(0, lA0, lB0, lA1, lB1, A, B);

    // Phase 2: Compute stencil on ping buffer (buf_sel=0)
    compute(0, lA0, lB0, lA1, lB1);

    // Phase 3: Store ping buffer results back to global memory (buf_sel=0)
    store(0, lA0, lB0, lA1, lB1, A, B);
}

} // extern "C"