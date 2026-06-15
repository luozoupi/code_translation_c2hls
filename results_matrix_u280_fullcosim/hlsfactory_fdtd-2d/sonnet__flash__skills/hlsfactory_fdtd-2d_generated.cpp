#include "fdtd-2d.h"

extern "C" {

void kernel_fdtd_2d(
		    double ex[ NX + 0][NY + 0],
		    double ey[ NX + 0][NY + 0],
		    double hz[ NX + 0][NY + 0],
		    double _fict_[ TMAX + 0])
{
#pragma HLS INTERFACE m_axi port=ex      offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=ey      offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=hz      offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=_fict_  offset=slave bundle=gmem3
#pragma HLS INTERFACE s_axilite port=ex      bundle=control
#pragma HLS INTERFACE s_axilite port=ey      bundle=control
#pragma HLS INTERFACE s_axilite port=hz      bundle=control
#pragma HLS INTERFACE s_axilite port=_fict_  bundle=control
#pragma HLS INTERFACE s_axilite port=return  bundle=control

    const int tmax = TMAX;
    const int nx   = NX;
    const int ny   = NY;

    // Local buffers for stencil computation to enable pipelining
    double l_ex[NX][NY];
    double l_ey[NX][NY];
    double l_hz[NX][NY];
    double l_fict[TMAX];

#pragma HLS ARRAY_PARTITION variable=l_ex  cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_ey  cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_hz  cyclic factor=8 dim=2

    // Load _fict_
    for (int t = 0; t < tmax; t++) {
#pragma HLS PIPELINE II=1
        l_fict[t] = _fict_[t];
    }

    // Load ex from global memory
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
#pragma HLS PIPELINE II=1
            l_ex[i][j] = ex[i][j];
        }
    }

    // Load ey from global memory
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
#pragma HLS PIPELINE II=1
            l_ey[i][j] = ey[i][j];
        }
    }

    // Load hz from global memory
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
#pragma HLS PIPELINE II=1
            l_hz[i][j] = hz[i][j];
        }
    }

    // Main time-step loop
    for (int t = 0; t < tmax; t++) {

        // Phase 1: update ey[0][j] = _fict_[t]
        for (int j = 0; j < ny; j++) {
#pragma HLS PIPELINE II=1
            l_ey[0][j] = l_fict[t];
        }

        // Phase 2: update ey[i][j] for i >= 1
        for (int i = 1; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
#pragma HLS PIPELINE II=1
                l_ey[i][j] = l_ey[i][j] - 0.5 * (l_hz[i][j] - l_hz[i-1][j]);
            }
        }

        // Phase 3: update ex[i][j] for j >= 1
        for (int i = 0; i < nx; i++) {
            for (int j = 1; j < ny; j++) {
#pragma HLS PIPELINE II=1
                l_ex[i][j] = l_ex[i][j] - 0.5 * (l_hz[i][j] - l_hz[i][j-1]);
            }
        }

        // Phase 4: update hz[i][j]
        for (int i = 0; i < nx - 1; i++) {
            for (int j = 0; j < ny - 1; j++) {
#pragma HLS PIPELINE II=1
                l_hz[i][j] = l_hz[i][j] - 0.7 * (l_ex[i][j+1] - l_ex[i][j] +
                                                    l_ey[i+1][j] - l_ey[i][j]);
            }
        }
    }

    // Write back ex to global memory
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
#pragma HLS PIPELINE II=1
            ex[i][j] = l_ex[i][j];
        }
    }

    // Write back ey to global memory
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
#pragma HLS PIPELINE II=1
            ey[i][j] = l_ey[i][j];
        }
    }

    // Write back hz to global memory
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
#pragma HLS PIPELINE II=1
            hz[i][j] = l_hz[i][j];
        }
    }
}

} // extern "C"