#include "mvt.h"
#include <string.h>


#define TILE 256

void kernel_mvt(
		double x1[ N + 0],
		double x2[ N + 0],
		double y_1[ N + 0],
		double y_2[ N + 0],
		double A[ N + 0][N + 0])
{
#pragma HLS INLINE off

    const int n = N;

    // ---- Local buffers ----
    double y1_local[N];
    double y2_local[N];
    double x1_local[N];
    double x2_local[N];

    // ---- LOAD vectors (reused across all rows) ----
    load_y1:
    for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
        y1_local[j] = y_1[j];
    }
    load_y2:
    for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
        y2_local[j] = y_2[j];
    }
    load_x1:
    for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        x1_local[i] = x1[i];
    }
    load_x2:
    for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        x2_local[i] = x2[i];
    }

    // ============================================================
    // First loop: x1[i] += sum_j A[i][j] * y_1[j]
    // Process A row-tiles
    // ============================================================
    loop1_tile:
    for (int it = 0; it < n; it += TILE) {
        int rows = (it + TILE <= n) ? TILE : (n - it);

        // LOAD tile of A rows into local buffer
        static double A_tile[TILE][N];
        load_A1:
        for (int r = 0; r < rows; r++) {
            for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
                A_tile[r][j] = A[it + r][j];
            }
        }

        // COMPUTE on local buffers
        compute1:
        for (int r = 0; r < rows; r++) {
            double acc1 = x1_local[it + r];
            loop1_j:
            for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
                acc1 = acc1 + A_tile[r][j] * y1_local[j];
            }
            x1_local[it + r] = acc1;
        }
    }

    // ============================================================
    // Second loop: x2[i] += sum_j A[j][i] * y_2[j]
    // Process column-tiles of A (i.e., output index i tiled)
    // ============================================================
    loop2_tile:
    for (int it = 0; it < n; it += TILE) {
        int cols = (it + TILE <= n) ? TILE : (n - it);

        // LOAD tile of A columns into local buffer (A[j][it..it+cols])
        static double A_tile2[N][TILE];
        load_A2:
        for (int j = 0; j < n; j++) {
            for (int c = 0; c < cols; c++) {
#pragma HLS PIPELINE II=1
                A_tile2[j][c] = A[j][it + c];
            }
        }

        // COMPUTE on local buffers
        compute2:
        for (int c = 0; c < cols; c++) {
            double acc2 = x2_local[it + c];
            loop2_j:
            for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
                acc2 = acc2 + A_tile2[j][c] * y2_local[j];
            }
            x2_local[it + c] = acc2;
        }
    }

    // ---- STORE results ----
    store_x1:
    for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        x1[i] = x1_local[i];
    }
    store_x2:
    for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        x2[i] = x2_local[i];
    }
}


extern "C" {
void workload(
		double x1[ N + 0],
		double x2[ N + 0],
		double y_1[ N + 0],
		double y_2[ N + 0],
		double A[ N + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=x1  offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=x2  offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=y_1 offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=y_2 offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=A   offset=slave bundle=gmem4

#pragma HLS INTERFACE s_axilite port=x1  bundle=control
#pragma HLS INTERFACE s_axilite port=x2  bundle=control
#pragma HLS INTERFACE s_axilite port=y_1 bundle=control
#pragma HLS INTERFACE s_axilite port=y_2 bundle=control
#pragma HLS INTERFACE s_axilite port=A   bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    kernel_mvt(x1, x2, y_1, y_2, A);
}
}