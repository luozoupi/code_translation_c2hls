#include "mvt.h"
#include <string.h>

#ifndef TILE
#define TILE 32
#endif

extern "C" {

void kernel_mvt(
		double x1[ N + 0],
		double x2[ N + 0],
		double y_1[ N + 0],
		double y_2[ N + 0],
		double A[ N + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=x1  offset=slave bundle=gmem0 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=x2  offset=slave bundle=gmem1 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=y_1 offset=slave bundle=gmem2 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=y_2 offset=slave bundle=gmem3 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=A   offset=slave bundle=gmem4 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE s_axilite port=x1      bundle=control
#pragma HLS INTERFACE s_axilite port=x2      bundle=control
#pragma HLS INTERFACE s_axilite port=y_1     bundle=control
#pragma HLS INTERFACE s_axilite port=y_2     bundle=control
#pragma HLS INTERFACE s_axilite port=A       bundle=control
#pragma HLS INTERFACE s_axilite port=return  bundle=control

    const int n = N;

    // -------------------------------------------------------
    // Loop 1: x1[i] += A[i][j] * y_1[j]
    // -------------------------------------------------------

    // Double buffers for loop 1
    double local_y1[2][TILE];
#pragma HLS ARRAY_PARTITION variable=local_y1 cyclic factor=8 dim=2

    double local_A1[2][TILE][TILE];
#pragma HLS ARRAY_PARTITION variable=local_A1 cyclic factor=8 dim=3

    loop1_i_tile: for (int i_start = 0; i_start < n; i_start += TILE) {
#pragma HLS LOOP_TRIPCOUNT min=(N+TILE-1)/TILE max=(N+TILE-1)/TILE

        int i_tile = (i_start + TILE <= n) ? TILE : (n - i_start);

        double local_x1[TILE];
#pragma HLS ARRAY_PARTITION variable=local_x1 cyclic factor=8 dim=1

        // Load x1 tile
        for (int i = 0; i < i_tile; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
            local_x1[i] = x1[i_start + i];
        }

        // Pre-load first tile into ping buffer (index 0)
        {
            int j_tile0 = (TILE <= n) ? TILE : n;
            for (int j = 0; j < j_tile0; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
                local_y1[0][j] = y_1[j];
            }
            for (int i = 0; i < i_tile; i++) {
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
                for (int j = 0; j < j_tile0; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
                    local_A1[0][i][j] = A[i_start + i][j];
                }
            }
        }

        loop1_j_tile: for (int j_start = 0; j_start < n; j_start += TILE) {
#pragma HLS LOOP_TRIPCOUNT min=(N+TILE-1)/TILE max=(N+TILE-1)/TILE

            int j_tile = (j_start + TILE <= n) ? TILE : (n - j_start);
            int ping = (j_start / TILE) % 2;
            int pong = 1 - ping;

            // Load NEXT tile into pong buffer
            int j_next = j_start + TILE;
            if (j_next < n) {
                int j_tile_next = (j_next + TILE <= n) ? TILE : (n - j_next);
                for (int j = 0; j < j_tile_next; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
                    local_y1[pong][j] = y_1[j_next + j];
                }
                for (int i = 0; i < i_tile; i++) {
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
                    for (int j = 0; j < j_tile_next; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
                        local_A1[pong][i][j] = A[i_start + i][j_next + j];
                    }
                }
            }

            // Compute from ping buffer
            for (int i = 0; i < i_tile; i++) {
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
                double sum = local_x1[i];
                for (int j = 0; j < j_tile; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
                    sum += local_A1[ping][i][j] * local_y1[ping][j];
                }
                local_x1[i] = sum;
            }
        }

        // Store x1 tile
        for (int i = 0; i < i_tile; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
            x1[i_start + i] = local_x1[i];
        }
    }

    // -------------------------------------------------------
    // Loop 2: x2[i] += A[j][i] * y_2[j]
    // -------------------------------------------------------

    // Double buffers for loop 2
    double local_y2[2][TILE];
#pragma HLS ARRAY_PARTITION variable=local_y2 cyclic factor=8 dim=2

    double local_A2[2][TILE][TILE];
#pragma HLS ARRAY_PARTITION variable=local_A2 cyclic factor=8 dim=3

    loop2_i_tile: for (int i_start = 0; i_start < n; i_start += TILE) {
#pragma HLS LOOP_TRIPCOUNT min=(N+TILE-1)/TILE max=(N+TILE-1)/TILE

        int i_tile = (i_start + TILE <= n) ? TILE : (n - i_start);

        double local_x2[TILE];
#pragma HLS ARRAY_PARTITION variable=local_x2 cyclic factor=8 dim=1

        // Load x2 tile
        for (int i = 0; i < i_tile; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
            local_x2[i] = x2[i_start + i];
        }

        // Pre-load first tile into ping buffer (index 0)
        {
            int j_tile0 = (TILE <= n) ? TILE : n;
            for (int j = 0; j < j_tile0; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
                local_y2[0][j] = y_2[j];
            }
            for (int j = 0; j < j_tile0; j++) {
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
                for (int i = 0; i < i_tile; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
                    local_A2[0][j][i] = A[j][i_start + i];
                }
            }
        }

        loop2_j_tile: for (int j_start = 0; j_start < n; j_start += TILE) {
#pragma HLS LOOP_TRIPCOUNT min=(N+TILE-1)/TILE max=(N+TILE-1)/TILE

            int j_tile = (j_start + TILE <= n) ? TILE : (n - j_start);
            int ping = (j_start / TILE) % 2;
            int pong = 1 - ping;

            // Load NEXT tile into pong buffer
            int j_next = j_start + TILE;
            if (j_next < n) {
                int j_tile_next = (j_next + TILE <= n) ? TILE : (n - j_next);
                for (int j = 0; j < j_tile_next; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
                    local_y2[pong][j] = y_2[j_next + j];
                }
                for (int j = 0; j < j_tile_next; j++) {
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
                    for (int i = 0; i < i_tile; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
                        local_A2[pong][j][i] = A[j_next + j][i_start + i];
                    }
                }
            }

            // Compute from ping buffer
            for (int i = 0; i < i_tile; i++) {
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
                double sum = local_x2[i];
                for (int j = 0; j < j_tile; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
                    sum += local_A2[ping][j][i] * local_y2[ping][j];
                }
                local_x2[i] = sum;
            }
        }

        // Store x2 tile
        for (int i = 0; i < i_tile; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=TILE max=TILE
            x2[i_start + i] = local_x2[i];
        }
    }
}

} // extern "C"