#include "trmm.h"

// Tile sizes
#define TILE_M 16
#define TILE_N 16

extern "C" {

void kernel_trmm(
        double alpha,
        double A[M + 0][M + 0],
        double B[M + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=alpha bundle=control
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local tile buffers for A and B tiles
    double lA[TILE_M][TILE_M];
    double lB_tile[TILE_M][TILE_N];
    double lBk_tile[TILE_M][TILE_N];

#pragma HLS ARRAY_PARTITION variable=lA complete dim=1
#pragma HLS ARRAY_PARTITION variable=lA complete dim=2
#pragma HLS ARRAY_PARTITION variable=lB_tile complete dim=2
#pragma HLS ARRAY_PARTITION variable=lBk_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=lBk_tile complete dim=2

    // Process output B in tiles of TILE_M rows x TILE_N columns
    for (int i0 = 0; i0 < M; i0 += TILE_M) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=4
        int i_end = (i0 + TILE_M < M) ? (i0 + TILE_M) : M;
        int ti_size = i_end - i0;

        for (int j0 = 0; j0 < N; j0 += TILE_N) {
#pragma HLS LOOP_TRIPCOUNT min=5 max=5
            int j_end = (j0 + TILE_N < N) ? (j0 + TILE_N) : N;
            int tj_size = j_end - j0;

            // ---- LOAD: Load B[i0..i_end][j0..j_end] into lB_tile ----
            load_B:
            for (int ti = 0; ti < TILE_M; ti++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
                for (int tj = 0; tj < TILE_N; tj++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
                    if (ti < ti_size && tj < tj_size)
                        lB_tile[ti][tj] = B[i0 + ti][j0 + tj];
                    else
                        lB_tile[ti][tj] = 0.0;
                }
            }

            // ---- COMPUTE: accumulate contributions from all k-tiles ----
            // For each row i in [i0, i_end), sum over k in (i, M):
            //   B[i][j] += A[k][i] * B[k][j]
            // We accumulate into lB_tile (starting with loaded values)

            for (int k0 = 0; k0 < M; k0 += TILE_M) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=4
                int k_end = (k0 + TILE_M < M) ? (k0 + TILE_M) : M;
                int tk_size = k_end - k0;

                // Load B[k0..k_end][j0..j_end] into lBk_tile
                load_Bk:
                for (int tk = 0; tk < TILE_M; tk++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
                    for (int tj = 0; tj < TILE_N; tj++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
                        if (tk < tk_size && tj < tj_size)
                            lBk_tile[tk][tj] = B[k0 + tk][j0 + tj];
                        else
                            lBk_tile[tk][tj] = 0.0;
                    }
                }

                // Load A[k0..k_end][i0..i_end] into lA
                // (we need A[k][i], i.e., column i of A => row i of lower-triangular)
                load_A:
                for (int tk = 0; tk < TILE_M; tk++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
                    for (int ti = 0; ti < TILE_M; ti++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
                        if (tk < tk_size && ti < ti_size)
                            lA[tk][ti] = A[k0 + tk][i0 + ti];
                        else
                            lA[tk][ti] = 0.0;
                    }
                }

                // Accumulate: for each (i, j) in tile, sum over k in k-tile
                // Only where k > i (strict lower triangular contribution)
                compute:
                for (int ti = 0; ti < TILE_M; ti++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
                    if (ti >= ti_size) continue;
                    int global_i = i0 + ti;
                    for (int tj = 0; tj < TILE_N; tj++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS DEPENDENCE variable=lB_tile inter false
                        if (tj >= tj_size) continue;
                        double acc = 0.0;
                        for (int tk = 0; tk < TILE_M; tk++) {
#pragma HLS UNROLL
                            int global_k = k0 + tk;
                            // Only accumulate if global_k > global_i
                            if (tk < tk_size && global_k > global_i) {
                                acc += lA[tk][ti] * lBk_tile[tk][tj];
                            }
                        }
                        lB_tile[ti][tj] += acc;
                    }
                }
            }

            // Apply alpha scaling to the accumulated lB_tile
            scale:
            for (int ti = 0; ti < TILE_M; ti++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
                for (int tj = 0; tj < TILE_N; tj++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
                    lB_tile[ti][tj] = alpha * lB_tile[ti][tj];
                }
            }

            // ---- STORE: Write lB_tile back to B[i0..i_end][j0..j_end] ----
            store_B:
            for (int ti = 0; ti < TILE_M; ti++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
                for (int tj = 0; tj < TILE_N; tj++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
                    if (ti < ti_size && tj < tj_size)
                        B[i0 + ti][j0 + tj] = lB_tile[ti][tj];
                }
            }
        }
    }
}

} // extern "C"