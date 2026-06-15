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

    double lA[TILE_M][TILE_M];
    double lBk[TILE_M][TILE_N];
    double lB[TILE_M][TILE_N];

#pragma HLS ARRAY_PARTITION variable=lA cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=lBk cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=lB cyclic factor=4 dim=2

    for (int i0 = 0; i0 < M; i0 += TILE_M) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=4
        int i_end = (i0 + TILE_M < M) ? (i0 + TILE_M) : M;
        int ti_size = i_end - i0;

        for (int j0 = 0; j0 < N; j0 += TILE_N) {
#pragma HLS LOOP_TRIPCOUNT min=5 max=5
            int j_end = (j0 + TILE_N < N) ? (j0 + TILE_N) : N;
            int tj_size = j_end - j0;

            // Load B tile
            load_B:
            for (int ti = 0; ti < TILE_M; ti++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
                for (int tj = 0; tj < TILE_N; tj++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
                    if (ti < ti_size && tj < tj_size)
                        lB[ti][tj] = B[i0 + ti][j0 + tj];
                    else
                        lB[ti][tj] = 0.0;
                }
            }

            // Accumulate over k-tiles
            for (int k0 = 0; k0 < M; k0 += TILE_M) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=4
                int k_end = (k0 + TILE_M < M) ? (k0 + TILE_M) : M;
                int tk_size = k_end - k0;

                // Load A tile: A[k0..k_end][i0..i_end]
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

                // Load Bk tile: B[k0..k_end][j0..j_end]
                load_Bk:
                for (int tk = 0; tk < TILE_M; tk++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
                    for (int tj = 0; tj < TILE_N; tj++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
                        if (tk < tk_size && tj < tj_size)
                            lBk[tk][tj] = B[k0 + tk][j0 + tj];
                        else
                            lBk[tk][tj] = 0.0;
                    }
                }

                // Compute: lB[ti][tj] += sum_tk A[k0+tk][i0+ti] * B[k0+tk][j0+tj]
                // Only when global_k > global_i (upper triangular)
                compute:
                for (int ti = 0; ti < TILE_M; ti++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
                    if (ti >= ti_size) continue;
                    int global_i = i0 + ti;
                    for (int tj = 0; tj < TILE_N; tj++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
                        if (tj >= tj_size) continue;
                        double acc = 0.0;
                        for (int tk = 0; tk < TILE_M; tk++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
                            int global_k = k0 + tk;
                            if (tk < tk_size && global_k > global_i) {
                                acc += lA[tk][ti] * lBk[tk][tj];
                            }
                        }
                        lB[ti][tj] += acc;
                    }
                }
            } // end k0

            // Scale and store
            store_B:
            for (int ti = 0; ti < TILE_M; ti++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
                for (int tj = 0; tj < TILE_N; tj++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
                    if (ti < ti_size && tj < tj_size)
                        B[i0 + ti][j0 + tj] = alpha * lB[ti][tj];
                }
            }
        }
    }
}

} // extern "C"