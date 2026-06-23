#include "fdtd-2d.h"
#include <cstring>

// Load the fict value for one time step into the selected ping-pong buffer
static void load_fict(double fict_buf[2], const double _fict_[TMAX], int t, int flag)
{
    if (t < TMAX) {
        fict_buf[flag] = _fict_[t];
    }
}

// Compute one full time step using the fict value in the selected buffer
static void compute_step(
    double ex_buf[NX][NY],
    double ey_buf[NX][NY],
    double hz_buf[NX][NY],
    double fict_buf[2],
    int flag)
{
    int i, j;
    double fict_val = fict_buf[flag];

    for (j = 0; j < NY; j++) {
#pragma HLS LOOP_TRIPCOUNT min=NY max=NY
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
        ey_buf[0][j] = fict_val;
    }
    for (i = 1; i < NX; i++)
        for (j = 0; j < NY; j++) {
#pragma HLS LOOP_TRIPCOUNT min=NY max=NY
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
            ey_buf[i][j] = ey_buf[i][j] - 0.5*(hz_buf[i][j]-hz_buf[i-1][j]);
        }
    for (i = 0; i < NX; i++)
        for (j = 1; j < NY; j++) {
#pragma HLS LOOP_TRIPCOUNT min=NY max=NY
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
            ex_buf[i][j] = ex_buf[i][j] - 0.5*(hz_buf[i][j]-hz_buf[i][j-1]);
        }
    for (i = 0; i < NX - 1; i++)
        for (j = 0; j < NY - 1; j++) {
#pragma HLS LOOP_TRIPCOUNT min=NY max=NY
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
            hz_buf[i][j] = hz_buf[i][j] - 0.7*  (ex_buf[i][j+1] - ex_buf[i][j] +
                                                 ey_buf[i+1][j] - ey_buf[i][j]);
        }
}


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

    // Ping-pong (double) buffer for the per-time-step fict value, so the load
    // of fict for step t+1 can overlap the compute of step t.
    double fict_buf[2];
#pragma HLS ARRAY_PARTITION variable=fict_buf complete dim=1

    // Partition along the column dimension (dim=2) with factor matching the
    // unroll factor (4) so that unrolled iterations can access their data in
    // parallel. cyclic banking provides simultaneous access to consecutive
    // columns [j], [j+1], [j+2], [j+3].
#pragma HLS ARRAY_PARTITION variable=ex_buf cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=ey_buf cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=hz_buf cyclic factor=4 dim=2

    // ---- Load entire grids into local buffers ----
load_ex:
    for (i = 0; i < nx; i++)
        for (j = 0; j < ny; j++) {
#pragma HLS LOOP_TRIPCOUNT min=NY max=NY
#pragma HLS PIPELINE II=1
            ex_buf[i][j] = ex[i][j];
        }
load_ey:
    for (i = 0; i < nx; i++)
        for (j = 0; j < ny; j++) {
#pragma HLS LOOP_TRIPCOUNT min=NY max=NY
#pragma HLS PIPELINE II=1
            ey_buf[i][j] = ey[i][j];
        }
load_hz:
    for (i = 0; i < nx; i++)
        for (j = 0; j < ny; j++) {
#pragma HLS LOOP_TRIPCOUNT min=NY max=NY
#pragma HLS PIPELINE II=1
            hz_buf[i][j] = hz[i][j];
        }

    // ---- Compute phase with double buffering on the fict value ----
    // Prologue: load fict for the first time step into buffer 0.
    load_fict(fict_buf, _fict_, 0, 0);

compute_time:
    for (t = 0; t < tmax; t++)
    {
#pragma HLS LOOP_TRIPCOUNT min=TMAX max=TMAX
        int flag = t & 1;          // buffer currently holding fict for step t
        int next_flag = flag ^ 1;  // buffer to prefetch fict for step t+1

        // Overlap: load fict for the next step while computing the current one.
        load_fict(fict_buf, _fict_, t + 1, next_flag);
        compute_step(ex_buf, ey_buf, hz_buf, fict_buf, flag);
    }

    // ---- Store results back to global memory ----
store_ex:
    for (i = 0; i < nx; i++)
        for (j = 0; j < ny; j++) {
#pragma HLS LOOP_TRIPCOUNT min=NY max=NY
#pragma HLS PIPELINE II=1
            ex[i][j] = ex_buf[i][j];
        }
store_ey:
    for (i = 0; i < nx; i++)
        for (j = 0; j < ny; j++) {
#pragma HLS LOOP_TRIPCOUNT min=NY max=NY
#pragma HLS PIPELINE II=1
            ey[i][j] = ey_buf[i][j];
        }
store_hz:
    for (i = 0; i < nx; i++)
        for (j = 0; j < ny; j++) {
#pragma HLS LOOP_TRIPCOUNT min=NY max=NY
#pragma HLS PIPELINE II=1
            hz[i][j] = hz_buf[i][j];
        }

}