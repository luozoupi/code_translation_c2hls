#include "durbin.h"

#define TILE_SIZE 32

static void load_r(double r_global[N], double r_local[N]) {
    for (int tile = 0; tile < N; tile += TILE_SIZE) {
        int tile_end = (tile + TILE_SIZE < N) ? TILE_SIZE : (N - tile);
        double r_tile[TILE_SIZE];
        #pragma HLS ARRAY_PARTITION variable=r_tile complete dim=1
        for (int i = 0; i < tile_end; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_SIZE
            r_tile[i] = r_global[tile + i];
        }
        for (int i = 0; i < tile_end; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_SIZE
            r_local[tile + i] = r_tile[i];
        }
    }
}

static void compute_durbin(double r_local[N], double y_local[N]) {
    double z[N];
    #pragma HLS ARRAY_PARTITION variable=z cyclic factor=4 dim=1

    double alpha;
    double beta;
    double sum;

    y_local[0] = -r_local[0];
    beta  =  1.0;
    alpha = -r_local[0];

    for (int k = 1; k < N; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=N-1
        beta = (1 - alpha * alpha) * beta;

        sum = 0.0;
        for (int i = 0; i < k; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS UNROLL factor=4
            #pragma HLS LOOP_TRIPCOUNT min=1 max=N-1
            #pragma HLS DEPENDENCE variable=r_local inter false
            #pragma HLS DEPENDENCE variable=y_local inter false
            sum += r_local[k - i - 1] * y_local[i];
        }
        alpha = -(r_local[k] + sum) / beta;

        // Process y update in tiles of TILE_SIZE
        for (int tile = 0; tile < k; tile += TILE_SIZE) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=N/TILE_SIZE
            int tile_end = (tile + TILE_SIZE < k) ? (tile + TILE_SIZE) : k;
            // Load tile of y_local into local tile buffer
            double y_tile_fwd[TILE_SIZE];
            double y_tile_rev[TILE_SIZE];
            #pragma HLS ARRAY_PARTITION variable=y_tile_fwd cyclic factor=4 dim=1
            #pragma HLS ARRAY_PARTITION variable=y_tile_rev cyclic factor=4 dim=1

            for (int i = tile; i < tile_end; i++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS UNROLL factor=4
                #pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_SIZE
                #pragma HLS DEPENDENCE variable=y_local inter false
                y_tile_fwd[i - tile] = y_local[i];
                y_tile_rev[i - tile] = y_local[k - i - 1];
            }
            for (int i = tile; i < tile_end; i++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS UNROLL factor=4
                #pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_SIZE
                #pragma HLS DEPENDENCE variable=z inter false
                z[i] = y_tile_fwd[i - tile] + alpha * y_tile_rev[i - tile];
            }
        }

        for (int tile = 0; tile < k; tile += TILE_SIZE) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=N/TILE_SIZE
            int tile_end = (tile + TILE_SIZE < k) ? (tile + TILE_SIZE) : k;
            double z_tile[TILE_SIZE];
            #pragma HLS ARRAY_PARTITION variable=z_tile cyclic factor=4 dim=1

            for (int i = tile; i < tile_end; i++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS UNROLL factor=4
                #pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_SIZE
                #pragma HLS DEPENDENCE variable=z inter false
                z_tile[i - tile] = z[i];
            }
            for (int i = tile; i < tile_end; i++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS UNROLL factor=4
                #pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_SIZE
                #pragma HLS DEPENDENCE variable=y_local inter false
                y_local[i] = z_tile[i - tile];
            }
        }

        y_local[k] = alpha;
    }
}

static void store_y(double y_local[N], double y_global[N]) {
    for (int tile = 0; tile < N; tile += TILE_SIZE) {
        int tile_end = (tile + TILE_SIZE < N) ? TILE_SIZE : (N - tile);
        double y_tile[TILE_SIZE];
        #pragma HLS ARRAY_PARTITION variable=y_tile complete dim=1

        for (int i = 0; i < tile_end; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_SIZE
            y_tile[i] = y_local[tile + i];
        }
        for (int i = 0; i < tile_end; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_SIZE
            y_global[tile + i] = y_tile[i];
        }
    }
}

extern "C" {

void kernel_durbin(
		   double r[ N + 0],
		   double y[ N + 0])
{
#pragma HLS INTERFACE m_axi port=r offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=y offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=r bundle=control
#pragma HLS INTERFACE s_axilite port=y bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local buffers
    double r_local[N];
    double y_local[N];

#pragma HLS ARRAY_PARTITION variable=r_local cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=y_local cyclic factor=4 dim=1

    // Phase 1: Load r from global memory in tiles
    load_r(r, r_local);

    // Phase 2: Compute Durbin algorithm on local buffers
    compute_durbin(r_local, y_local);

    // Phase 3: Store y to global memory in tiles
    store_y(y_local, y);
}

} // extern "C"