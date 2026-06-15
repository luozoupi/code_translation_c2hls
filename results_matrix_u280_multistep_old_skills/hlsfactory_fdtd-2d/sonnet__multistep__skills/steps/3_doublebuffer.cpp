#include "fdtd-2d.h"
#include <string.h>

#define TILE_I 8

extern "C" {

void kernel_fdtd_2d(
		    
		    
		    double ex[ NX + 0][NY + 0],
		    double ey[ NX + 0][NY + 0],
		    double hz[ NX + 0][NY + 0],
		    double _fict_[ TMAX + 0])
{
#pragma HLS INTERFACE m_axi port=ex     offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=ey     offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=hz     offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=_fict_ offset=slave bundle=gmem3
#pragma HLS INTERFACE s_axilite port=ex      bundle=control
#pragma HLS INTERFACE s_axilite port=ey      bundle=control
#pragma HLS INTERFACE s_axilite port=hz      bundle=control
#pragma HLS INTERFACE s_axilite port=_fict_  bundle=control
#pragma HLS INTERFACE s_axilite port=return  bundle=control

    const int tmax = TMAX;
    const int nx = NX;
    const int ny = NY;

    // Double-buffered local tile buffers: 2 ping-pong copies
    double l_ex_0[TILE_I + 1][NY];
    double l_ex_1[TILE_I + 1][NY];
    double l_ey_0[TILE_I + 1][NY];
    double l_ey_1[TILE_I + 1][NY];
    double l_hz_0[TILE_I + 1][NY];
    double l_hz_1[TILE_I + 1][NY];

#pragma HLS ARRAY_PARTITION variable=l_ex_0 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_ex_1 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_ey_0 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_ey_1 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_hz_0 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_hz_1 cyclic factor=8 dim=2

    // Local buffer for _fict_
    double l_fict[TMAX];
#pragma HLS ARRAY_PARTITION variable=l_fict complete dim=1

    // --- Load _fict_ ---
    load_fict: for (int t = 0; t < tmax; t++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=TMAX max=TMAX
        l_fict[t] = _fict_[t];
    }

    int t, i, j;

    for (t = 0; t < tmax; t++)
    {
#pragma HLS LOOP_TRIPCOUNT min=TMAX max=TMAX

        // =========================================================
        // Phase 1: Update ey[0][j] = _fict_[t]  (boundary row only)
        // =========================================================
        {
            // Load tile rows [0 .. min(TILE_I, nx)-1] for ey and hz into buffer 0
            load_ey0_tile: for (int ii = 0; ii < TILE_I && ii < nx; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=TILE_I max=TILE_I
                for (int j = 0; j < ny; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=NY max=NY
#pragma HLS DEPENDENCE variable=l_ey_0 inter false
#pragma HLS DEPENDENCE variable=l_hz_0 inter false
#pragma HLS UNROLL factor=8
                    l_ey_0[ii][j] = ey[ii][j];
                    l_hz_0[ii][j] = hz[ii][j];
                }
            }
            // Boundary: ey[0][j] = _fict_[t]
            set_ey0: for (j = 0; j < ny; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=NY max=NY
#pragma HLS DEPENDENCE variable=l_ey_0 inter false
#pragma HLS UNROLL factor=8
                l_ey_0[0][j] = l_fict[t];
            }
            // Store back ey row 0 only
            store_ey0: for (j = 0; j < ny; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=NY max=NY
#pragma HLS UNROLL factor=8
                ey[0][j] = l_ey_0[0][j];
            }
        }

        // =========================================================
        // Phase 2: Update ey[i][j] for i=1..nx-1, tiled over rows
        //          Double-buffered: load tile k+1 while computing tile k
        // =========================================================
        {
            // Number of tiles
            int num_tiles_ey = 0;
            for (int i0 = 1; i0 < nx; i0 += TILE_I) num_tiles_ey++;

            // Prefetch tile 0 into buffer 0
            {
                int i0 = 1;
                int i_end = (i0 + TILE_I < nx) ? i0 + TILE_I : nx;
                int tile_rows = i_end - i0;
                load_ey_pre: for (int ii = 0; ii < tile_rows + 1 && (i0 - 1 + ii) < nx; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_I+1
                    for (int j = 0; j < ny; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=NY max=NY
#pragma HLS DEPENDENCE variable=l_hz_0 inter false
#pragma HLS DEPENDENCE variable=l_ey_0 inter false
#pragma HLS UNROLL factor=8
                        l_hz_0[ii][j] = hz[i0 - 1 + ii][j];
                        if (ii < tile_rows)
                            l_ey_0[ii][j] = ey[i0 + ii][j];
                    }
                }
            }

            int ping = 0; // 0: compute from buf0, load into buf1; 1: compute from buf1, load into buf0
            int tile_idx = 0;
            update_ey_tiles: for (int i0 = 1; i0 < nx; i0 += TILE_I)
            {
#pragma HLS LOOP_TRIPCOUNT min=NX/TILE_I max=NX/TILE_I
                int i_end = (i0 + TILE_I < nx) ? i0 + TILE_I : nx;
                int tile_rows = i_end - i0;

                // Next tile parameters
                int i0_next = i0 + TILE_I;
                int i_end_next = (i0_next + TILE_I < nx) ? i0_next + TILE_I : nx;
                int tile_rows_next = i_end_next - i0_next;
                bool has_next = (i0_next < nx);

                if (ping == 0) {
                    // Compute from buf0
                    compute_ey_p0: for (int ii = 0; ii < tile_rows; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_I
                        for (int j = 0; j < ny; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=NY max=NY
#pragma HLS DEPENDENCE variable=l_ey_0 inter false
#pragma HLS DEPENDENCE variable=l_hz_0 inter false
#pragma HLS UNROLL factor=8
                            l_ey_0[ii][j] = l_ey_0[ii][j] - 0.5 * (l_hz_0[ii + 1][j] - l_hz_0[ii][j]);
                        }
                    }
                    // Store buf0
                    store_ey_p0: for (int ii = 0; ii < tile_rows; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_I
                        for (int j = 0; j < ny; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=NY max=NY
#pragma HLS DEPENDENCE variable=l_ey_0 inter false
#pragma HLS UNROLL factor=8
                            ey[i0 + ii][j] = l_ey_0[ii][j];
                        }
                    }
                    // Load next tile into buf1
                    if (has_next) {
                        load_ey_p0: for (int ii = 0; ii < tile_rows_next + 1 && (i0_next - 1 + ii) < nx; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_I+1
                            for (int j = 0; j < ny; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=NY max=NY
#pragma HLS DEPENDENCE variable=l_hz_1 inter false
#pragma HLS DEPENDENCE variable=l_ey_1 inter false
#pragma HLS UNROLL factor=8
                                l_hz_1[ii][j] = hz[i0_next - 1 + ii][j];
                                if (ii < tile_rows_next)
                                    l_ey_1[ii][j] = ey[i0_next + ii][j];
                            }
                        }
                    }
                } else {
                    // Compute from buf1
                    compute_ey_p1: for (int ii = 0; ii < tile_rows; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_I
                        for (int j = 0; j < ny; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=NY max=NY
#pragma HLS DEPENDENCE variable=l_ey_1 inter false
#pragma HLS DEPENDENCE variable=l_hz_1 inter false
#pragma HLS UNROLL factor=8
                            l_ey_1[ii][j] = l_ey_1[ii][j] - 0.5 * (l_hz_1[ii + 1][j] - l_hz_1[ii][j]);
                        }
                    }
                    // Store buf1
                    store_ey_p1: for (int ii = 0; ii < tile_rows; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_I
                        for (int j = 0; j < ny; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=NY max=NY
#pragma HLS DEPENDENCE variable=l_ey_1 inter false
#pragma HLS UNROLL factor=8
                            ey[i0 + ii][j] = l_ey_1[ii][j];
                        }
                    }
                    // Load next tile into buf0
                    if (has_next) {
                        load_ey_p1: for (int ii = 0; ii < tile_rows_next + 1 && (i0_next - 1 + ii) < nx; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_I+1
                            for (int j = 0; j < ny; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=NY max=NY
#pragma HLS DEPENDENCE variable=l_hz_0 inter false
#pragma HLS DEPENDENCE variable=l_ey_0 inter false
#pragma HLS UNROLL factor=8
                                l_hz_0[ii][j] = hz[i0_next - 1 + ii][j];
                                if (ii < tile_rows_next)
                                    l_ey_0[ii][j] = ey[i0_next + ii][j];
                            }
                        }
                    }
                }
                ping = 1 - ping;
                tile_idx++;
            }
        }

        // =========================================================
        // Phase 3: Update ex[i][j] for j=1..ny-1, tiled over rows
        //          Double-buffered
        // =========================================================
        {
            // Prefetch tile 0 into buffer 0
            {
                int i0 = 0;
                int i_end = (i0 + TILE_I < nx) ? i0 + TILE_I : nx;
                int tile_rows = i_end - i0;
                load_ex_pre: for (int ii = 0; ii < tile_rows; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_I
                    for (int j = 0; j < ny; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=NY max=NY
#pragma HLS DEPENDENCE variable=l_ex_0 inter false
#pragma HLS DEPENDENCE variable=l_hz_0 inter false
#pragma HLS UNROLL factor=8
                        l_ex_0[ii][j] = ex[i0 + ii][j];
                        l_hz_0[ii][j] = hz[i0 + ii][j];
                    }
                }
            }

            int ping = 0;
            update_ex_tiles: for (int i0 = 0; i0 < nx; i0 += TILE_I)
            {
#pragma HLS LOOP_TRIPCOUNT min=NX/TILE_I max=NX/TILE_I
                int i_end = (i0 + TILE_I < nx) ? i0 + TILE_I : nx;
                int tile_rows = i_end - i0;

                int i0_next = i0 + TILE_I;
                int i_end_next = (i0_next + TILE_I < nx) ? i0_next + TILE_I : nx;
                int tile_rows_next = i_end_next - i0_next;
                bool has_next = (i0_next < nx);

                if (ping == 0) {
                    // Compute from buf0
                    compute_ex_p0: for (int ii = 0; ii < tile_rows; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_I
                        for (int j = 1; j < ny; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=NY-1 max=NY-1
#pragma HLS DEPENDENCE variable=l_ex_0 inter false
#pragma HLS DEPENDENCE variable=l_hz_0 inter false
#pragma HLS UNROLL factor=8
                            l_ex_0[ii][j] = l_ex_0[ii][j] - 0.5 * (l_hz_0[ii][j] - l_hz_0[ii][j - 1]);
                        }
                    }
                    // Store buf0
                    store_ex_p0: for (int ii = 0; ii < tile_rows; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_I
                        for (int j = 0; j < ny; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=NY max=NY
#pragma HLS DEPENDENCE variable=l_ex_0 inter false
#pragma HLS UNROLL factor=8
                            ex[i0 + ii][j] = l_ex_0[ii][j];
                        }
                    }
                    // Load next tile into buf1
                    if (has_next) {
                        load_ex_p0: for (int ii = 0; ii < tile_rows_next; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_I
                            for (int j = 0; j < ny; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=NY max=NY
#pragma HLS DEPENDENCE variable=l_ex_1 inter false
#pragma HLS DEPENDENCE variable=l_hz_1 inter false
#pragma HLS UNROLL factor=8
                                l_ex_1[ii][j] = ex[i0_next + ii][j];
                                l_hz_1[ii][j] = hz[i0_next + ii][j];
                            }
                        }
                    }
                } else {
                    // Compute from buf1
                    compute_ex_p1: for (int ii = 0; ii < tile_rows; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_I
                        for (int j = 1; j < ny; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=NY-1 max=NY-1
#pragma HLS DEPENDENCE variable=l_ex_1 inter false
#pragma HLS DEPENDENCE variable=l_hz_1 inter false
#pragma HLS UNROLL factor=8
                            l_ex_1[ii][j] = l_ex_1[ii][j] - 0.5 * (l_hz_1[ii][j] - l_hz_1[ii][j - 1]);
                        }
                    }
                    // Store buf1
                    store_ex_p1: for (int ii = 0; ii < tile_rows; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_I
                        for (int j = 0; j < ny; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=NY max=NY
#pragma HLS DEPENDENCE variable=l_ex_1 inter false
#pragma HLS UNROLL factor=8
                            ex[i0 + ii][j] = l_ex_1[ii][j];
                        }
                    }
                    // Load next tile into buf0
                    if (has_next) {
                        load_ex_p1: for (int ii = 0; ii < tile_rows_next; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_I
                            for (int j = 0; j < ny; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=NY max=NY
#pragma HLS DEPENDENCE variable=l_ex_0 inter false
#pragma HLS DEPENDENCE variable=l_hz_0 inter false
#pragma HLS UNROLL factor=8
                                l_ex_0[ii][j] = ex[i0_next + ii][j];
                                l_hz_0[ii][j] = hz[i0_next + ii][j];
                            }
                        }
                    }
                }
                ping = 1 - ping;
            }
        }

        // =========================================================
        // Phase 4: Update hz[i][j], tiled over rows, double-buffered
        // =========================================================
        {
            // Prefetch tile 0 into buffer 0
            {
                int i0 = 0;
                int i_end = (i0 + TILE_I < nx - 1) ? i0 + TILE_I : nx - 1;
                int tile_rows = i_end - i0;
                load_hz_pre: for (int ii = 0; ii < tile_rows + 1; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_I+1
                    for (int j = 0; j < ny; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=NY max=NY
#pragma HLS DEPENDENCE variable=l_ey_0 inter false
#pragma HLS DEPENDENCE variable=l_ex_0 inter false
#pragma HLS DEPENDENCE variable=l_hz_0 inter false
#pragma HLS UNROLL factor=8
                        l_ey_0[ii][j] = ey[i0 + ii][j];
                        if (ii < tile_rows) {
                            l_ex_0[ii][j] = ex[i0 + ii][j];
                            l_hz_0[ii][j] = hz[i0 + ii][j];
                        }
                    }
                }
            }

            int ping = 0;
            update_hz_tiles: for (int i0 = 0; i0 < nx - 1; i0 += TILE_I)
            {
#pragma HLS LOOP_TRIPCOUNT min=NX/TILE_I max=NX/TILE_I
                int i_end = (i0 + TILE_I < nx - 1) ? i0 + TILE_I : nx - 1;
                int tile_rows = i_end - i0;

                int i0_next = i0 + TILE_I;
                int i_end_next = (i0_next + TILE_I < nx - 1) ? i0_next + TILE_I : nx - 1;
                int tile_rows_next = i_end_next - i0_next;
                bool has_next = (i0_next < nx - 1);

                if (ping == 0) {
                    // Compute from buf0
                    compute_hz_p0: for (int ii = 0; ii < tile_rows; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_I
                        for (int j = 0; j < ny - 1; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=NY-1 max=NY-1
#pragma HLS DEPENDENCE variable=l_hz_0 inter false
#pragma HLS DEPENDENCE variable=l_ex_0 inter false
#pragma HLS DEPENDENCE variable=l_ey_0 inter false
#pragma HLS UNROLL factor=8
                            l_hz_0[ii][j] = l_hz_0[ii][j] - 0.7 * (
                                l_ex_0[ii][j + 1] - l_ex_0[ii][j] +
                                l_ey_0[ii + 1][j] - l_ey_0[ii][j]);
                        }
                    }
                    // Store buf0
                    store_hz_p0: for (int ii = 0; ii < tile_rows; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_I
                        for (int j = 0; j < ny - 1; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=NY-1 max=NY-1
#pragma HLS DEPENDENCE variable=l_hz_0 inter false
#pragma HLS UNROLL factor=8
                            hz[i0 + ii][j] = l_hz_0[ii][j];
                        }
                    }
                    // Load next tile into buf1
                    if (has_next) {
                        load_hz_p0: for (int ii = 0; ii < tile_rows_next + 1; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_I+1
                            for (int j = 0; j < ny; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=NY max=NY
#pragma HLS DEPENDENCE variable=l_ey_1 inter false
#pragma HLS DEPENDENCE variable=l_ex_1 inter false
#pragma HLS DEPENDENCE variable=l_hz_1 inter false
#pragma HLS UNROLL factor=8
                                l_ey_1[ii][j] = ey[i0_next + ii][j];
                                if (ii < tile_rows_next) {
                                    l_ex_1[ii][j] = ex[i0_next + ii][j];
                                    l_hz_1[ii][j] = hz[i0_next + ii][j];
                                }
                            }
                        }
                    }
                } else {
                    // Compute from buf1
                    compute_hz_p1: for (int ii = 0; ii < tile_rows; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_I
                        for (int j = 0; j < ny - 1; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=NY-1 max=NY-1
#pragma HLS DEPENDENCE variable=l_hz_1 inter false
#pragma HLS DEPENDENCE variable=l_ex_1 inter false
#pragma HLS DEPENDENCE variable=l_ey_1 inter false
#pragma HLS UNROLL factor=8
                            l_hz_1[ii][j] = l_hz_1[ii][j] - 0.7 * (
                                l_ex_1[ii][j + 1] - l_ex_1[ii][j] +
                                l_ey_1[ii + 1][j] - l_ey_1[ii][j]);
                        }
                    }
                    // Store buf1
                    store_hz_p1: for (int ii = 0; ii < tile_rows; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_I
                        for (int j = 0; j < ny - 1; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=NY-1 max=NY-1
#pragma HLS DEPENDENCE variable=l_hz_1 inter false
#pragma HLS UNROLL factor=8
                            hz[i0 + ii][j] = l_hz_1[ii][j];
                        }
                    }
                    // Load next tile into buf0
                    if (has_next) {
                        load_hz_p1: for (int ii = 0; ii < tile_rows_next + 1; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_I+1
                            for (int j = 0; j < ny; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=NY max=NY
#pragma HLS DEPENDENCE variable=l_ey_0 inter false
#pragma HLS DEPENDENCE variable=l_ex_0 inter false
#pragma HLS DEPENDENCE variable=l_hz_0 inter false
#pragma HLS UNROLL factor=8
                                l_ey_0[ii][j] = ey[i0_next + ii][j];
                                if (ii < tile_rows_next) {
                                    l_ex_0[ii][j] = ex[i0_next + ii][j];
                                    l_hz_0[ii][j] = hz[i0_next + ii][j];
                                }
                            }
                        }
                    }
                }
                ping = 1 - ping;
            }
        }

    } // end time loop
}

} // extern "C"