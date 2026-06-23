#include "durbin.h"
#include <cstring>

#define TILE 256

void kernel_durbin(
		   double r[ N + 0],
		   double y[ N + 0])
{
#pragma HLS INTERFACE m_axi port=r offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=y offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=r bundle=control
#pragma HLS INTERFACE s_axilite port=y bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;

    double r_local[N];
    double y_local[N];
    double z[N];
#pragma HLS ARRAY_PARTITION variable=r_local cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=y_local cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=z cyclic factor=8 dim=1

    double alpha;
    double beta;
    double sum;

    int i, k, t;

    // ---- LOAD PHASE: stage input r into local memory in tiles ----
LOAD_TILE:
    for (t = 0; t < n; t += TILE) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=1
        int chunk = (n - t < TILE) ? (n - t) : TILE;
    LOAD_INNER:
        for (i = 0; i < chunk; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
#pragma HLS PIPELINE II=1
            r_local[t + i] = r[t + i];
        }
    }

    // ---- COMPUTE PHASE: operate entirely on local buffers ----
    y_local[0] = -r_local[0];
    beta = 1.0;
    alpha = -r_local[0];

COMPUTE:
    for (k = 1; k < n; k++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=119
        beta = (1 - alpha * alpha) * beta;

        sum = 0.0;
    SUM_LOOP:
        for (i = 0; i < k; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=119
#pragma HLS PIPELINE II=1
            sum += r_local[k - i - 1] * y_local[i];
        }

        alpha = -(r_local[k] + sum) / beta;

    UPDATE_Z:
        for (i = 0; i < k; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=119
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS DEPENDENCE variable=z inter false
#pragma HLS DEPENDENCE variable=y_local inter false
            z[i] = y_local[i] + alpha * y_local[k - i - 1];
        }

    COPY_Y:
        for (i = 0; i < k; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=119
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS DEPENDENCE variable=y_local inter false
#pragma HLS DEPENDENCE variable=z inter false
            y_local[i] = z[i];
        }

        y_local[k] = alpha;
    }

    // ---- STORE PHASE: write local results back to global memory in tiles ----
STORE_TILE:
    for (t = 0; t < n; t += TILE) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=1
        int chunk = (n - t < TILE) ? (n - t) : TILE;
    STORE_INNER:
        for (i = 0; i < chunk; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
#pragma HLS PIPELINE II=1
            y[t + i] = y_local[t + i];
        }
    }
}