#include "mvt.h"
#include <string.h>


#define TILE 60
#define UF 4

// ---- Double-buffered LOAD of A row-tile (loop 1) ----
static void load_A1(double A[N + 0][N + 0], double A_buf1[TILE][N],
                    double A_buf2[TILE][N], int it, int rows, int flag)
{
#pragma HLS INLINE off
    if (flag == 0) {
        for (int r = 0; r < rows; r++) {
#pragma HLS LOOP_TRIPCOUNT min=60 max=60
            for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
                A_buf1[r][j] = A[it + r][j];
            }
        }
    } else {
        for (int r = 0; r < rows; r++) {
#pragma HLS LOOP_TRIPCOUNT min=60 max=60
            for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
                A_buf2[r][j] = A[it + r][j];
            }
        }
    }
}

// ---- Double-buffered COMPUTE of loop 1 ----
static void compute_A1(double A_buf1[TILE][N], double A_buf2[TILE][N],
                       double x1_local[N], double y1_local[N],
                       int it, int rows, int flag)
{
#pragma HLS INLINE off
    if (flag == 0) {
        for (int r = 0; r < rows; r++) {
#pragma HLS LOOP_TRIPCOUNT min=60 max=60
            double acc1 = x1_local[it + r];
            for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=UF
                acc1 = acc1 + A_buf1[r][j] * y1_local[j];
            }
            x1_local[it + r] = acc1;
        }
    } else {
        for (int r = 0; r < rows; r++) {
#pragma HLS LOOP_TRIPCOUNT min=60 max=60
            double acc1 = x1_local[it + r];
            for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=UF
                acc1 = acc1 + A_buf2[r][j] * y1_local[j];
            }
            x1_local[it + r] = acc1;
        }
    }
}

// ---- Double-buffered LOAD of A column-tile (loop 2) ----
static void load_A2(double A[N + 0][N + 0], double A_buf1[N][TILE],
                    double A_buf2[N][TILE], int it, int cols, int flag)
{
#pragma HLS INLINE off
    if (flag == 0) {
        for (int j = 0; j < N; j++) {
#pragma HLS LOOP_TRIPCOUNT min=120 max=120
            for (int c = 0; c < cols; c++) {
#pragma HLS PIPELINE II=1
                A_buf1[j][c] = A[j][it + c];
            }
        }
    } else {
        for (int j = 0; j < N; j++) {
#pragma HLS LOOP_TRIPCOUNT min=120 max=120
            for (int c = 0; c < cols; c++) {
#pragma HLS PIPELINE II=1
                A_buf2[j][c] = A[j][it + c];
            }
        }
    }
}

// ---- Double-buffered COMPUTE of loop 2 ----
static void compute_A2(double A_buf1[N][TILE], double A_buf2[N][TILE],
                       double x2_local[N], double y2_local[N],
                       int it, int cols, int flag)
{
#pragma HLS INLINE off
    if (flag == 0) {
        for (int c = 0; c < cols; c++) {
#pragma HLS LOOP_TRIPCOUNT min=60 max=60
            double acc2 = x2_local[it + c];
            for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=UF
                acc2 = acc2 + A_buf1[j][c] * y2_local[j];
            }
            x2_local[it + c] = acc2;
        }
    } else {
        for (int c = 0; c < cols; c++) {
#pragma HLS LOOP_TRIPCOUNT min=60 max=60
            double acc2 = x2_local[it + c];
            for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=UF
                acc2 = acc2 + A_buf2[j][c] * y2_local[j];
            }
            x2_local[it + c] = acc2;
        }
    }
}

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
#pragma HLS ARRAY_PARTITION variable=y1_local cyclic factor=UF dim=1
#pragma HLS ARRAY_PARTITION variable=y2_local cyclic factor=UF dim=1

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
    // Double-buffered row tiling
    // ============================================================
    static double A1_buf1[TILE][N];
    static double A1_buf2[TILE][N];
#pragma HLS ARRAY_PARTITION variable=A1_buf1 cyclic factor=UF dim=2
#pragma HLS ARRAY_PARTITION variable=A1_buf2 cyclic factor=UF dim=2

    {
        // Number of tiles
        int num_tiles = (n + TILE - 1) / TILE;

        // Prologue: load first tile into buffer 0
        int it0 = 0;
        int rows0 = (it0 + TILE <= n) ? TILE : (n - it0);
        load_A1(A, A1_buf1, A1_buf2, it0, rows0, 0);

        loop1_tile:
        for (int t = 1; t < num_tiles; t++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=2
            int flag_load = t & 1;          // buffer to load tile t
            int flag_comp = (t - 1) & 1;    // buffer to compute tile t-1

            int it_load = t * TILE;
            int rows_load = (it_load + TILE <= n) ? TILE : (n - it_load);

            int it_comp = (t - 1) * TILE;
            int rows_comp = (it_comp + TILE <= n) ? TILE : (n - it_comp);

            // Load tile t while computing tile t-1 (overlap)
            load_A1(A, A1_buf1, A1_buf2, it_load, rows_load, flag_load);
            compute_A1(A1_buf1, A1_buf2, x1_local, y1_local, it_comp, rows_comp, flag_comp);
        }

        // Epilogue: compute last tile
        int t_last = num_tiles - 1;
        int flag_last = t_last & 1;
        int it_last = t_last * TILE;
        int rows_last = (it_last + TILE <= n) ? TILE : (n - it_last);
        compute_A1(A1_buf1, A1_buf2, x1_local, y1_local, it_last, rows_last, flag_last);
    }

    // ============================================================
    // Second loop: x2[i] += sum_j A[j][i] * y_2[j]
    // Double-buffered column tiling
    // ============================================================
    static double A2_buf1[N][TILE];
    static double A2_buf2[N][TILE];
#pragma HLS ARRAY_PARTITION variable=A2_buf1 cyclic factor=UF dim=1
#pragma HLS ARRAY_PARTITION variable=A2_buf2 cyclic factor=UF dim=1

    {
        int num_tiles = (n + TILE - 1) / TILE;

        // Prologue: load first column tile into buffer 0
        int it0 = 0;
        int cols0 = (it0 + TILE <= n) ? TILE : (n - it0);
        load_A2(A, A2_buf1, A2_buf2, it0, cols0, 0);

        loop2_tile:
        for (int t = 1; t < num_tiles; t++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=2
            int flag_load = t & 1;
            int flag_comp = (t - 1) & 1;

            int it_load = t * TILE;
            int cols_load = (it_load + TILE <= n) ? TILE : (n - it_load);

            int it_comp = (t - 1) * TILE;
            int cols_comp = (it_comp + TILE <= n) ? TILE : (n - it_comp);

            load_A2(A, A2_buf1, A2_buf2, it_load, cols_load, flag_load);
            compute_A2(A2_buf1, A2_buf2, x2_local, y2_local, it_comp, cols_comp, flag_comp);
        }

        // Epilogue: compute last tile
        int t_last = num_tiles - 1;
        int flag_last = t_last & 1;
        int it_last = t_last * TILE;
        int cols_last = (it_last + TILE <= n) ? TILE : (n - it_last);
        compute_A2(A2_buf1, A2_buf2, x2_local, y2_local, it_last, cols_last, flag_last);
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