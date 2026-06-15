#include "durbin.h"

#define TILE_SIZE 32

static void load_r(double r_global[N], double r_local[N]) {
    // Double-buffered tile loading: two tile buffers, alternate between them
    double r_tile_0[TILE_SIZE];
    double r_tile_1[TILE_SIZE];
    #pragma HLS ARRAY_PARTITION variable=r_tile_0 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=r_tile_1 complete dim=1

    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    // Pre-load the first tile into buffer 0
    {
        int tile_end_0 = (TILE_SIZE < N) ? TILE_SIZE : N;
        for (int i = 0; i < tile_end_0; i++) {
            #pragma HLS PIPELINE II=1
            r_tile_0[i] = r_global[i];
        }
    }

    for (int t = 0; t < num_tiles; t++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=(N+TILE_SIZE-1)/TILE_SIZE
        int tile_start = t * TILE_SIZE;
        int tile_end = (tile_start + TILE_SIZE < N) ? TILE_SIZE : (N - tile_start);

        // Pre-load next tile into the alternate buffer (if it exists)
        int next_t = t + 1;
        if (next_t < num_tiles) {
            int next_tile_start = next_t * TILE_SIZE;
            int next_tile_end = (next_tile_start + TILE_SIZE < N) ? TILE_SIZE : (N - next_tile_start);
            if (t % 2 == 0) {
                // Next tile goes into buffer 1
                for (int i = 0; i < next_tile_end; i++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_SIZE
                    r_tile_1[i] = r_global[next_tile_start + i];
                }
            } else {
                // Next tile goes into buffer 0
                for (int i = 0; i < next_tile_end; i++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_SIZE
                    r_tile_0[i] = r_global[next_tile_start + i];
                }
            }
        }

        // Write current tile from the current buffer to r_local
        if (t % 2 == 0) {
            for (int i = 0; i < tile_end; i++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_SIZE
                r_local[tile_start + i] = r_tile_0[i];
            }
        } else {
            for (int i = 0; i < tile_end; i++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_SIZE
                r_local[tile_start + i] = r_tile_1[i];
            }
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
    // Double-buffered tile store: two tile buffers, alternate between them
    double y_tile_0[TILE_SIZE];
    double y_tile_1[TILE_SIZE];
    #pragma HLS ARRAY_PARTITION variable=y_tile_0 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=y_tile_1 complete dim=1

    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    // Pre-fill buffer 0 with first tile from y_local
    {
        int tile_end_0 = (TILE_SIZE < N) ? TILE_SIZE : N;
        for (int i = 0; i < tile_end_0; i++) {
            #pragma HLS PIPELINE II=1
            y_tile_0[i] = y_local[i];
        }
    }

    for (int t = 0; t < num_tiles; t++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=(N+TILE_SIZE-1)/TILE_SIZE
        int tile_start = t * TILE_SIZE;
        int tile_end = (tile_start + TILE_SIZE < N) ? TILE_SIZE : (N - tile_start);

        // Pre-load next tile into the alternate buffer (if it exists)
        int next_t = t + 1;
        if (next_t < num_tiles) {
            int next_tile_start = next_t * TILE_SIZE;
            int next_tile_end = (next_tile_start + TILE_SIZE < N) ? TILE_SIZE : (N - next_tile_start);
            if (t % 2 == 0) {
                // Next tile goes into buffer 1
                for (int i = 0; i < next_tile_end; i++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_SIZE
                    y_tile_1[i] = y_local[next_tile_start + i];
                }
            } else {
                // Next tile goes into buffer 0
                for (int i = 0; i < next_tile_end; i++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_SIZE
                    y_tile_0[i] = y_local[next_tile_start + i];
                }
            }
        }

        // Write current buffer to global memory
        if (t % 2 == 0) {
            for (int i = 0; i < tile_end; i++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_SIZE
                y_global[tile_start + i] = y_tile_0[i];
            }
        } else {
            for (int i = 0; i < tile_end; i++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_SIZE
                y_global[tile_start + i] = y_tile_1[i];
            }
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

    // Phase 1: Load r from global memory in double-buffered tiles
    load_r(r, r_local);

    // Phase 2: Compute Durbin algorithm on local buffers
    compute_durbin(r_local, y_local);

    // Phase 3: Store y to global memory in double-buffered tiles
    store_y(y_local, y);
}

} // extern "C"