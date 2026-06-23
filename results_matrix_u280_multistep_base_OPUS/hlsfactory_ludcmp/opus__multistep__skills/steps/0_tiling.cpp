#include "ludcmp.h"
#include <cstring>

extern "C" {
void kernel_ludcmp(
		   double A[ N + 0][N + 0],
		   double b[ N + 0],
		   double x[ N + 0],
		   double y[ N + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=y offset=slave bundle=gmem3
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=b bundle=control
#pragma HLS INTERFACE s_axilite port=x bundle=control
#pragma HLS INTERFACE s_axilite port=y bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;
    const int TILE = 256;  // tile size for staging linear transfers

    int i, j, k;
    int t, tj;

    double w;

    // Local buffers to stage data from global memory for reuse during the
    // computation-heavy LU decomposition phase.
    static double A_local[N][N];
#pragma HLS ARRAY_PARTITION variable=A_local cyclic factor=8 dim=2
    static double b_local[N];
    static double x_local[N];
    static double y_local[N];

    // ---------------- LOAD PHASE ----------------
    // Load A from global memory into local buffer one row at a time, using
    // tiled bursts within each row to improve memory locality / throughput.
    for (i = 0; i < n; i++) {
        for (tj = 0; tj < n; tj += TILE) {
            int chunk = (tj + TILE <= n) ? TILE : (n - tj);
            // Stage a tile of this row into a temporary local buffer via burst.
            double row_tile[TILE];
#pragma HLS ARRAY_PARTITION variable=row_tile cyclic factor=8
            memcpy(row_tile, &A[i][tj], chunk * sizeof(double));
            for (j = 0; j < chunk; j++) {
#pragma HLS PIPELINE II=1
                A_local[i][tj + j] = row_tile[j];
            }
        }
    }

    // Load b from global memory into a local tile, then into b_local.
    for (t = 0; t < n; t += TILE) {
        int chunk = (t + TILE <= n) ? TILE : (n - t);
        double b_tile[TILE];
        memcpy(b_tile, &b[t], chunk * sizeof(double));
        for (j = 0; j < chunk; j++) {
#pragma HLS PIPELINE II=1
            b_local[t + j] = b_tile[j];
        }
    }

    // ---------------- COMPUTE PHASE ----------------
    // LU decomposition (in-place on local buffer).
    for (i = 0; i < n; i++) {
        for (j = 0; j < i; j++) {
            w = A_local[i][j];
            for (k = 0; k < j; k++) {
#pragma HLS PIPELINE II=1
                w -= A_local[i][k] * A_local[k][j];
            }
            A_local[i][j] = w / A_local[j][j];
        }
        for (j = i; j < n; j++) {
            w = A_local[i][j];
            for (k = 0; k < i; k++) {
#pragma HLS PIPELINE II=1
                w -= A_local[i][k] * A_local[k][j];
            }
            A_local[i][j] = w;
        }
    }

    // Forward substitution to compute y.
    for (i = 0; i < n; i++) {
        w = b_local[i];
        for (j = 0; j < i; j++) {
#pragma HLS PIPELINE II=1
            w -= A_local[i][j] * y_local[j];
        }
        y_local[i] = w;
    }

    // Backward substitution to compute x.
    for (i = n - 1; i >= 0; i--) {
        w = y_local[i];
        for (j = i + 1; j < n; j++) {
#pragma HLS PIPELINE II=1
            w -= A_local[i][j] * x_local[j];
        }
        x_local[i] = w / A_local[i][i];
    }

    // ---------------- STORE PHASE ----------------
    // Write A back to global memory using tiled bursts within each row.
    for (i = 0; i < n; i++) {
        for (tj = 0; tj < n; tj += TILE) {
            int chunk = (tj + TILE <= n) ? TILE : (n - tj);
            double row_tile[TILE];
#pragma HLS ARRAY_PARTITION variable=row_tile cyclic factor=8
            for (j = 0; j < chunk; j++) {
#pragma HLS PIPELINE II=1
                row_tile[j] = A_local[i][tj + j];
            }
            memcpy(&A[i][tj], row_tile, chunk * sizeof(double));
        }
    }

    // Write x and y back to global memory using tiled bursts.
    for (t = 0; t < n; t += TILE) {
        int chunk = (t + TILE <= n) ? TILE : (n - t);
        double x_tile[TILE];
        double y_tile[TILE];
        for (j = 0; j < chunk; j++) {
#pragma HLS PIPELINE II=1
            x_tile[j] = x_local[t + j];
            y_tile[j] = y_local[t + j];
        }
        memcpy(&x[t], x_tile, chunk * sizeof(double));
        memcpy(&y[t], y_tile, chunk * sizeof(double));
    }
}
}