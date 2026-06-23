#include "durbin.h"
#include <cstring>

#define TILE 256

// Load a tile of r[] into one of two ping-pong buffers selected by flag
static void load_tile(double r[N], double r_buf0[N], double r_buf1[N],
                      int t, int chunk, int flag)
{
LOAD_INNER:
    for (int i = 0; i < chunk; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
#pragma HLS PIPELINE II=1
        double v = r[t + i];
        if (flag == 0)
            r_buf0[t + i] = v;
        else
            r_buf1[t + i] = v;
    }
}

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

    // ---- DOUBLE-BUFFERED input staging buffers (ping-pong) ----
    double r_local_0[N];
    double r_local_1[N];
    double r_local[N];
    double y_local[N];
    double z[N];
#pragma HLS ARRAY_PARTITION variable=r_local_0 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=r_local_1 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=r_local cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=y_local cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=z cyclic factor=8 dim=1

    double alpha;
    double beta;
    double sum;

    int i, k, t;

    // ---- LOAD PHASE: stage input r into local memory with double buffering ----
    // Issue loads into alternating buffers so successive tile loads can overlap,
    // then merge the staged tiles into the working buffer.
    int tile_count = 0;
LOAD_TILE:
    for (t = 0; t < n; t += TILE) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=1
        int chunk = (n - t < TILE) ? (n - t) : TILE;
        int flag = (t / TILE) % 2;   // alternate ping-pong buffer per tile

        // Load current tile into the selected ping-pong buffer
        load_tile(r, r_local_0, r_local_1, t, chunk, flag);

        // Consume the just-loaded tile into the working buffer
    MERGE_INNER:
        for (i = 0; i < chunk; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
#pragma HLS PIPELINE II=1
            r_local[t + i] = (flag == 0) ? r_local_0[t + i] : r_local_1[t + i];
        }
        tile_count++;
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