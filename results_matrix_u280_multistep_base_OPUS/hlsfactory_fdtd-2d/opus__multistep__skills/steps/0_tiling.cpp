#include "fdtd-2d.h"
#include <cstring>


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
#pragma HLS INTERFACE s_axilite port=ex     bundle=control
#pragma HLS INTERFACE s_axilite port=ey     bundle=control
#pragma HLS INTERFACE s_axilite port=hz     bundle=control
#pragma HLS INTERFACE s_axilite port=_fict_ bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int tmax = TMAX;
    const int nx = NX;
    const int ny = NY;

    int t, i, j;

    // Local tile buffers for the full 2D grid (staged once per time step)
    double ex_buf[NX][NY];
    double ey_buf[NX][NY];
    double hz_buf[NX][NY];
    double fict_buf[TMAX];

    // ---- Load _fict_ once ----
    for (t = 0; t < tmax; t++) {
#pragma HLS PIPELINE II=1
        fict_buf[t] = _fict_[t];
    }

    // ---- Load entire grids into local buffers ----
load_ex:
    for (i = 0; i < nx; i++)
        for (j = 0; j < ny; j++) {
#pragma HLS PIPELINE II=1
            ex_buf[i][j] = ex[i][j];
        }
load_ey:
    for (i = 0; i < nx; i++)
        for (j = 0; j < ny; j++) {
#pragma HLS PIPELINE II=1
            ey_buf[i][j] = ey[i][j];
        }
load_hz:
    for (i = 0; i < nx; i++)
        for (j = 0; j < ny; j++) {
#pragma HLS PIPELINE II=1
            hz_buf[i][j] = hz[i][j];
        }

    // ---- Compute phase: operate entirely on local buffers ----
compute_time:
    for (t = 0; t < tmax; t++)
    {
        for (j = 0; j < ny; j++) {
#pragma HLS PIPELINE II=1
            ey_buf[0][j] = fict_buf[t];
        }
        for (i = 1; i < nx; i++)
            for (j = 0; j < ny; j++) {
#pragma HLS PIPELINE II=1
                ey_buf[i][j] = ey_buf[i][j] - 0.5*(hz_buf[i][j]-hz_buf[i-1][j]);
            }
        for (i = 0; i < nx; i++)
            for (j = 1; j < ny; j++) {
#pragma HLS PIPELINE II=1
                ex_buf[i][j] = ex_buf[i][j] - 0.5*(hz_buf[i][j]-hz_buf[i][j-1]);
            }
        for (i = 0; i < nx - 1; i++)
            for (j = 0; j < ny - 1; j++) {
#pragma HLS PIPELINE II=1
                hz_buf[i][j] = hz_buf[i][j] - 0.7*  (ex_buf[i][j+1] - ex_buf[i][j] +
                                                     ey_buf[i+1][j] - ey_buf[i][j]);
            }
    }

    // ---- Store results back to global memory ----
store_ex:
    for (i = 0; i < nx; i++)
        for (j = 0; j < ny; j++) {
#pragma HLS PIPELINE II=1
            ex[i][j] = ex_buf[i][j];
        }
store_ey:
    for (i = 0; i < nx; i++)
        for (j = 0; j < ny; j++) {
#pragma HLS PIPELINE II=1
            ey[i][j] = ey_buf[i][j];
        }
store_hz:
    for (i = 0; i < nx; i++)
        for (j = 0; j < ny; j++) {
#pragma HLS PIPELINE II=1
            hz[i][j] = hz_buf[i][j];
        }

}