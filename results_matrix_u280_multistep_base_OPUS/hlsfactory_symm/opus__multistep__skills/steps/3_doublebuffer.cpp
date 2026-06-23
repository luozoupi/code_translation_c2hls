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
    const int TILE = 256;

    int i, j, k;

    // Full buffers required: B_local is read across all rows (column-wise)
    // in the temp2 reduction, and C_local accumulates across all rows.
    static double C_local[M][N];
    static double B_local[M][N];
#pragma HLS ARRAY_PARTITION variable=C_local cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=B_local cyclic factor=8 dim=2

    // Double-buffered per-row staging for A's row i: A[i][*] is only used
    // for the current row's compute, so it can be ping-ponged. Loading
    // A row (i+1) overlaps with compute of row i.
    static double A_row_1[M];
    static double A_row_2[M];
#pragma HLS ARRAY_PARTITION variable=A_row_1 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=A_row_2 cyclic factor=8 dim=1

    // ---------------- LOAD C PHASE ----------------
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

    // ---------------- LOAD B PHASE ----------------
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

    // ---------------- COMPUTE PHASE WITH DOUBLE-BUFFERED A ROW ----------
    // Prologue: load A row 0 into buffer set 1.
    LOAD_A_ROW0:for (k = 0; k < m; k++) {
#pragma HLS PIPELINE II=1
        A_row_1[k] = A[0][k];
    }

    COMP_I:for (i = 0; i < m; i++) {
        // flag selects which buffer holds the CURRENT row i.
        // i even -> current = A_row_1, next loads into A_row_2
        // i odd  -> current = A_row_2, next loads into A_row_1
        int flag = i % 2;

        // Load NEXT A row (i+1) into the opposite buffer, overlapping
        // with the compute below.
        if (i + 1 < m) {
            if (flag == 0) {
                LOAD_A_NEXT0:for (k = 0; k < m; k++) {
#pragma HLS PIPELINE II=1
                    A_row_2[k] = A[i + 1][k];
                }
            } else {
                LOAD_A_NEXT1:for (k = 0; k < m; k++) {
#pragma HLS PIPELINE II=1
                    A_row_1[k] = A[i + 1][k];
                }
            }
        }

        // Compute current row i using the current A-row buffer.
        if (flag == 0) {
            COMP_J0:for (j = 0; j < n; j++) {
                double temp2 = 0;
                COMP_K0:for (k = 0; k < i; k++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=M
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS DEPENDENCE variable=C_local inter false
                    C_local[k][j] += alpha * B_local[i][j] * A_row_1[k];
                    temp2 += B_local[k][j] * A_row_1[k];
                }
                C_local[i][j] = beta * C_local[i][j]
                                + alpha * B_local[i][j] * A_row_1[i]
                                + alpha * temp2;
            }
        } else {
            COMP_J1:for (j = 0; j < n; j++) {
                double temp2 = 0;
                COMP_K1:for (k = 0; k < i; k++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=M
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS DEPENDENCE variable=C_local inter false
                    C_local[k][j] += alpha * B_local[i][j] * A_row_2[k];
                    temp2 += B_local[k][j] * A_row_2[k];
                }
                C_local[i][j] = beta * C_local[i][j]
                                + alpha * B_local[i][j] * A_row_2[i]
                                + alpha * temp2;
            }
        }
    }

    // ---------------- STORE PHASE ----------------
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