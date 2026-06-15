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

    // Double-buffered local tile buffers for A and Bk tiles
    double lA_0[TILE_M][TILE_M];
    double lA_1[TILE_M][TILE_M];
    double lBk_tile_0[TILE_M][TILE_N];
    double lBk_tile_1[TILE_M][TILE_N];

    // Single buffer for B output tile (no overlap needed on this dimension)
    double lB_tile[TILE_M][TILE_N];

#pragma HLS ARRAY_PARTITION variable=lA_0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=lA_0 complete dim=2
#pragma HLS ARRAY_PARTITION variable=lA_1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=lA_1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=lBk_tile_0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=lBk_tile_0 complete dim=2
#pragma HLS ARRAY_PARTITION variable=lBk_tile_1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=lBk_tile_1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=lB_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=lB_tile complete dim=2

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
#pragma HLS UNROLL factor=4
                for (int tj = 0; tj < TILE_N; tj++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL factor=4
                    if (ti < ti_size && tj < tj_size)
                        lB_tile[ti][tj] = B[i0 + ti][j0 + tj];
                    else
                        lB_tile[ti][tj] = 0.0;
                }
            }

            // ---- PRE-LOAD: Load first k-tile (k0=0) into buffer set 0 ----
            {
                int k0_pre = 0;
                int k_end_pre = (k0_pre + TILE_M < M) ? (k0_pre + TILE_M) : M;
                int tk_size_pre = k_end_pre - k0_pre;

                // Pre-load lBk_tile_0 for k0=0
                preload_Bk:
                for (int tk = 0; tk < TILE_M; tk++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL factor=4
                    for (int tj = 0; tj < TILE_N; tj++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL factor=4
                        if (tk < tk_size_pre && tj < tj_size)
                            lBk_tile_0[tk][tj] = B[k0_pre + tk][j0 + tj];
                        else
                            lBk_tile_0[tk][tj] = 0.0;
                    }
                }

                // Pre-load lA_0 for k0=0
                preload_A:
                for (int tk = 0; tk < TILE_M; tk++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL factor=4
                    for (int ti = 0; ti < TILE_M; ti++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL factor=4
                        if (tk < tk_size_pre && ti < ti_size)
                            lA_0[tk][ti] = A[k0_pre + tk][i0 + ti];
                        else
                            lA_0[tk][ti] = 0.0;
                    }
                }
            }

            // ---- COMPUTE: accumulate contributions from all k-tiles ----
            // Double-buffer: buf=0 means compute from buffer_0, load into buffer_1
            //                buf=1 means compute from buffer_1, load into buffer_0
            for (int k0 = 0; k0 < M; k0 += TILE_M) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=4
                int k_end   = (k0 + TILE_M < M) ? (k0 + TILE_M) : M;
                int tk_size = k_end - k0;

                // Which buffer set to READ from this iteration
                int buf = (k0 / TILE_M) % 2;

                // ---- LOAD NEXT k-tile into the OTHER buffer (ping-pong) ----
                int k0_next = k0 + TILE_M;
                if (k0_next < M) {
                    int k_end_next  = (k0_next + TILE_M < M) ? (k0_next + TILE_M) : M;
                    int tk_size_next = k_end_next - k0_next;

                    if (buf == 0) {
                        // Load into buffer_1 while we compute from buffer_0
                        load_Bk_next_1:
                        for (int tk = 0; tk < TILE_M; tk++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL factor=4
                            for (int tj = 0; tj < TILE_N; tj++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL factor=4
                                if (tk < tk_size_next && tj < tj_size)
                                    lBk_tile_1[tk][tj] = B[k0_next + tk][j0 + tj];
                                else
                                    lBk_tile_1[tk][tj] = 0.0;
                            }
                        }
                        load_A_next_1:
                        for (int tk = 0; tk < TILE_M; tk++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL factor=4
                            for (int ti = 0; ti < TILE_M; ti++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL factor=4
                                if (tk < tk_size_next && ti < ti_size)
                                    lA_1[tk][ti] = A[k0_next + tk][i0 + ti];
                                else
                                    lA_1[tk][ti] = 0.0;
                            }
                        }
                    } else {
                        // Load into buffer_0 while we compute from buffer_1
                        load_Bk_next_0:
                        for (int tk = 0; tk < TILE_M; tk++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL factor=4
                            for (int tj = 0; tj < TILE_N; tj++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL factor=4
                                if (tk < tk_size_next && tj < tj_size)
                                    lBk_tile_0[tk][tj] = B[k0_next + tk][j0 + tj];
                                else
                                    lBk_tile_0[tk][tj] = 0.0;
                            }
                        }
                        load_A_next_0:
                        for (int tk = 0; tk < TILE_M; tk++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL factor=4
                            for (int ti = 0; ti < TILE_M; ti++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL factor=4
                                if (tk < tk_size_next && ti < ti_size)
                                    lA_0[tk][ti] = A[k0_next + tk][i0 + ti];
                                else
                                    lA_0[tk][ti] = 0.0;
                            }
                        }
                    }
                }

                // ---- COMPUTE from the current buffer (buf selects which) ----
                if (buf == 0) {
                    compute_0:
                    for (int ti = 0; ti < TILE_M; ti++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL factor=4
                        if (ti >= ti_size) continue;
                        int global_i = i0 + ti;
                        for (int tj = 0; tj < TILE_N; tj++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL factor=4
#pragma HLS DEPENDENCE variable=lB_tile inter false
                            if (tj >= tj_size) continue;
                            double acc = 0.0;
                            for (int tk = 0; tk < TILE_M; tk++) {
#pragma HLS UNROLL
                                int global_k = k0 + tk;
                                if (tk < tk_size && global_k > global_i) {
                                    acc += lA_0[tk][ti] * lBk_tile_0[tk][tj];
                                }
                            }
                            lB_tile[ti][tj] += acc;
                        }
                    }
                } else {
                    compute_1:
                    for (int ti = 0; ti < TILE_M; ti++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL factor=4
                        if (ti >= ti_size) continue;
                        int global_i = i0 + ti;
                        for (int tj = 0; tj < TILE_N; tj++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL factor=4
#pragma HLS DEPENDENCE variable=lB_tile inter false
                            if (tj >= tj_size) continue;
                            double acc = 0.0;
                            for (int tk = 0; tk < TILE_M; tk++) {
#pragma HLS UNROLL
                                int global_k = k0 + tk;
                                if (tk < tk_size && global_k > global_i) {
                                    acc += lA_1[tk][ti] * lBk_tile_1[tk][tj];
                                }
                            }
                            lB_tile[ti][tj] += acc;
                        }
                    }
                }
            } // end k0 loop

            // Apply alpha scaling to the accumulated lB_tile
            scale:
            for (int ti = 0; ti < TILE_M; ti++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL factor=4
                for (int tj = 0; tj < TILE_N; tj++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL factor=4
                    lB_tile[ti][tj] = alpha * lB_tile[ti][tj];
                }
            }

            // ---- STORE: Write lB_tile back to B[i0..i_end][j0..j_end] ----
            store_B:
            for (int ti = 0; ti < TILE_M; ti++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL factor=4
                for (int tj = 0; tj < TILE_N; tj++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL factor=4
                    if (ti < ti_size && tj < tj_size)
                        B[i0 + ti][j0 + tj] = lB_tile[ti][tj];
                }
            }
        }
    }
}

} // extern "C"