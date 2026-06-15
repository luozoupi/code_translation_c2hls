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

    // Local tile buffers: TILE_I+1 rows to accommodate halo/boundary
    double l_ex [TILE_I + 1][NY];
    double l_ey [TILE_I + 1][NY];
    double l_hz [TILE_I + 1][NY];

#pragma HLS ARRAY_PARTITION variable=l_ex  cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_ey  cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_hz  cyclic factor=8 dim=2

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
        // Tile containing row 0
        {
            // Load tile rows [0 .. min(TILE_I, nx)-1] for ey and hz
            load_ey0_tile: for (int ii = 0; ii < TILE_I && ii < nx; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=TILE_I max=TILE_I
                for (int j = 0; j < ny; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=NY max=NY
#pragma HLS DEPENDENCE variable=l_ey inter false
#pragma HLS DEPENDENCE variable=l_hz inter false
                    l_ey[ii][j] = ey[ii][j];
                    l_hz[ii][j] = hz[ii][j];
                }
            }
            // Boundary: ey[0][j] = _fict_[t]
            set_ey0: for (j = 0; j < ny; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=NY max=NY
#pragma HLS DEPENDENCE variable=l_ey inter false
                l_ey[0][j] = l_fict[t];
            }
            // Store back ey row 0 only
            store_ey0: for (j = 0; j < ny; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=NY max=NY
                ey[0][j] = l_ey[0][j];
            }
        }

        // =========================================================
        // Phase 2: Update ey[i][j] for i=1..nx-1, tiled over rows
        //          l_ey[i][j] -= 0.5*(hz[i][j] - hz[i-1][j])
        // =========================================================
        update_ey_tiles: for (int i0 = 1; i0 < nx; i0 += TILE_I)
        {
#pragma HLS LOOP_TRIPCOUNT min=NX/TILE_I max=NX/TILE_I
            int i_end = (i0 + TILE_I < nx) ? i0 + TILE_I : nx;
            int tile_rows = i_end - i0;  // number of rows in this tile

            // Load hz rows [i0-1 .. i_end-1] (need hz[i-1] for each i in tile)
            load_hz_ey: for (int ii = 0; ii < tile_rows + 1 && (i0 - 1 + ii) < nx; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_I+1
                for (int j = 0; j < ny; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=NY max=NY
#pragma HLS DEPENDENCE variable=l_hz inter false
#pragma HLS DEPENDENCE variable=l_ey inter false
                    l_hz[ii][j] = hz[i0 - 1 + ii][j];
                    if (ii < tile_rows)
                        l_ey[ii][j] = ey[i0 + ii][j];
                }
            }

            // Compute ey update within tile
            compute_ey: for (int ii = 0; ii < tile_rows; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_I
                for (int j = 0; j < ny; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=NY max=NY
#pragma HLS DEPENDENCE variable=l_ey inter false
#pragma HLS DEPENDENCE variable=l_hz inter false
                    // l_hz[ii] = hz[i0+ii-1], l_hz[ii+1] = hz[i0+ii]
                    l_ey[ii][j] = l_ey[ii][j] - 0.5 * (l_hz[ii + 1][j] - l_hz[ii][j]);
                }
            }

            // Store ey tile back
            store_ey: for (int ii = 0; ii < tile_rows; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_I
                for (int j = 0; j < ny; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=NY max=NY
#pragma HLS DEPENDENCE variable=l_ey inter false
                    ey[i0 + ii][j] = l_ey[ii][j];
                }
            }
        }

        // =========================================================
        // Phase 3: Update ex[i][j] for j=1..ny-1, tiled over rows
        //          l_ex[i][j] -= 0.5*(hz[i][j] - hz[i][j-1])
        // =========================================================
        update_ex_tiles: for (int i0 = 0; i0 < nx; i0 += TILE_I)
        {
#pragma HLS LOOP_TRIPCOUNT min=NX/TILE_I max=NX/TILE_I
            int i_end = (i0 + TILE_I < nx) ? i0 + TILE_I : nx;
            int tile_rows = i_end - i0;

            // Load ex and hz tiles
            load_ex_hz: for (int ii = 0; ii < tile_rows; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_I
                for (int j = 0; j < ny; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=NY max=NY
#pragma HLS DEPENDENCE variable=l_ex inter false
#pragma HLS DEPENDENCE variable=l_hz inter false
                    l_ex[ii][j] = ex[i0 + ii][j];
                    l_hz[ii][j] = hz[i0 + ii][j];
                }
            }

            // Compute ex update within tile
            compute_ex: for (int ii = 0; ii < tile_rows; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_I
                for (int j = 1; j < ny; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=NY-1 max=NY-1
#pragma HLS DEPENDENCE variable=l_ex inter false
#pragma HLS DEPENDENCE variable=l_hz inter false
                    l_ex[ii][j] = l_ex[ii][j] - 0.5 * (l_hz[ii][j] - l_hz[ii][j - 1]);
                }
            }

            // Store ex tile back
            store_ex: for (int ii = 0; ii < tile_rows; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_I
                for (int j = 0; j < ny; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=NY max=NY
#pragma HLS DEPENDENCE variable=l_ex inter false
                    ex[i0 + ii][j] = l_ex[ii][j];
                }
            }
        }

        // =========================================================
        // Phase 4: Update hz[i][j] for i=0..nx-2, j=0..ny-2, tiled
        //          hz[i][j] -= 0.7*(ex[i][j+1]-ex[i][j]+ey[i+1][j]-ey[i][j])
        // =========================================================
        update_hz_tiles: for (int i0 = 0; i0 < nx - 1; i0 += TILE_I)
        {
#pragma HLS LOOP_TRIPCOUNT min=NX/TILE_I max=NX/TILE_I
            int i_end = (i0 + TILE_I < nx - 1) ? i0 + TILE_I : nx - 1;
            int tile_rows = i_end - i0;

            // Load ex rows [i0..i_end-1], ey rows [i0..i_end], hz rows [i0..i_end-1]
            load_hz_update: for (int ii = 0; ii < tile_rows + 1; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_I+1
                for (int j = 0; j < ny; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=NY max=NY
#pragma HLS DEPENDENCE variable=l_ey inter false
#pragma HLS DEPENDENCE variable=l_ex inter false
#pragma HLS DEPENDENCE variable=l_hz inter false
                    l_ey[ii][j] = ey[i0 + ii][j];
                    if (ii < tile_rows) {
                        l_ex[ii][j] = ex[i0 + ii][j];
                        l_hz[ii][j] = hz[i0 + ii][j];
                    }
                }
            }

            // Compute hz update within tile
            compute_hz: for (int ii = 0; ii < tile_rows; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_I
                for (int j = 0; j < ny - 1; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=NY-1 max=NY-1
#pragma HLS DEPENDENCE variable=l_hz inter false
#pragma HLS DEPENDENCE variable=l_ex inter false
#pragma HLS DEPENDENCE variable=l_ey inter false
                    l_hz[ii][j] = l_hz[ii][j] - 0.7 * (
                        l_ex[ii][j + 1] - l_ex[ii][j] +
                        l_ey[ii + 1][j] - l_ey[ii][j]);
                }
            }

            // Store hz tile back
            store_hz: for (int ii = 0; ii < tile_rows; ii++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_I
                for (int j = 0; j < ny - 1; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=NY-1 max=NY-1
#pragma HLS DEPENDENCE variable=l_hz inter false
                    hz[i0 + ii][j] = l_hz[ii][j];
                }
            }
        }
    } // end time loop
}

} // extern "C"