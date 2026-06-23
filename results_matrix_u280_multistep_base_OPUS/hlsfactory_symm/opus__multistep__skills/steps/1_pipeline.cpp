#include "symm.h"
#include <cstring>

extern "C" {

void kernel_symm(
		 double alpha,
		 double beta,
		 double C[ M + 0][N + 0],
		 double A[ M + 0][M + 0],
		 double B[ M + 0][N + 0])
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

    const int m = M;
    const int n = N;

    const int TILE = 256; // tile size in elements per row-chunk copy

    int i, j, k;
    double temp2;

    // Stage the matrices into local on-chip buffers so the inner reduction
    // loops can be pipelined without competing for the same AXI port.
    static double C_local[M][N];
    static double A_local[M][M];
    static double B_local[M][N];
#pragma HLS ARRAY_PARTITION variable=A_local cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=C_local cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=B_local cyclic factor=8 dim=2

    // ---------------- LOAD PHASE ----------------
    // Load C in tiles of TILE elements (flattened row-major view).
    LOAD_C_I:for (i = 0; i < m; i++) {
        LOAD_C_J:for (j = 0; j < n; j += TILE) {
            int chunk = (n - j) < TILE ? (n - j) : TILE;
            LOAD_C_T:for (int t = 0; t < chunk; t++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
#pragma HLS PIPELINE II=1
                C_local[i][j + t] = C[i][j + t];
            }
        }
    }

    // Load A in tiles.
    LOAD_A_I:for (i = 0; i < m; i++) {
        LOAD_A_J:for (j = 0; j < m; j += TILE) {
            int chunk = (m - j) < TILE ? (m - j) : TILE;
            LOAD_A_T:for (int t = 0; t < chunk; t++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
#pragma HLS PIPELINE II=1
                A_local[i][j + t] = A[i][j + t];
            }
        }
    }

    // Load B in tiles.
    LOAD_B_I:for (i = 0; i < m; i++) {
        LOAD_B_J:for (j = 0; j < n; j += TILE) {
            int chunk = (n - j) < TILE ? (n - j) : TILE;
            LOAD_B_T:for (int t = 0; t < chunk; t++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
#pragma HLS PIPELINE II=1
                B_local[i][j + t] = B[i][j + t];
            }
        }
    }

    // ---------------- COMPUTE PHASE ----------------
    // Core computation operates entirely on local buffers.
    COMP_I:for (i = 0; i < m; i++) {
        COMP_J:for (j = 0; j < n; j++) {
            temp2 = 0;
            COMP_K:for (k = 0; k < i; k++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=M
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=C_local inter false
                C_local[k][j] += alpha * B_local[i][j] * A_local[i][k];
                temp2 += B_local[k][j] * A_local[i][k];
            }
            C_local[i][j] = beta * C_local[i][j]
                            + alpha * B_local[i][j] * A_local[i][i]
                            + alpha * temp2;
        }
    }

    // ---------------- STORE PHASE ----------------
    // Store C back to global memory in tiles.
    STORE_C_I:for (i = 0; i < m; i++) {
        STORE_C_J:for (j = 0; j < n; j += TILE) {
            int chunk = (n - j) < TILE ? (n - j) : TILE;
            STORE_C_T:for (int t = 0; t < chunk; t++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
#pragma HLS PIPELINE II=1
                C[i][j + t] = C_local[i][j + t];
            }
        }
    }
}

}