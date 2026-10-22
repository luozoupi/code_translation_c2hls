/* Naive Vitis HLS baseline of plain.cpp for SODA kernel 'sobel2d': same top/signature, m_axi per array + s_axilite,
 * no optimisation pragmas. NOTE: large static local buffer(s) (intermediate / ping-pong, up to 8 MB for 128^3 float)
 * are fine for csim but will map to BRAM/URAM (or fail to fit) in csynth; a real design must tile or stream. */
#include <math.h>
#include <stdint.h>
#include "sobel2d.h"

extern "C" {
void workload(const int16_t img[HEIGHT][WIDTH],
              uint16_t mag[HEIGHT][WIDTH])
{
#pragma HLS INTERFACE m_axi port=img offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=mag offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=img bundle=control
#pragma HLS INTERFACE s_axilite port=mag bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            if (x >= 1 && x < WIDTH - 1 && y >= 1 && y < HEIGHT - 1) {
                /* mag_x: int16 local; mag_y: uint16 local (truncate at the store, as SODA does) */
                int16_t mx = (int16_t)((img[y-1][x+1] - img[y-1][x-1]) + (img[y][x+1] - img[y][x-1]) * 3 + (img[y+1][x+1] - img[y+1][x-1]));
                uint16_t my = (uint16_t)((img[y+1][x-1] - img[y-1][x-1]) + (img[y+1][x] - img[y-1][x]) * 3 + (img[y+1][x+1] - img[y-1][x+1]));
                /* SODA: 65535 - (mx*mx + my*my) in int, stored uint16. Done in uint32 here (no signed overflow UB);
                   the low 16 bits are identical. */
                uint32_t s = (uint32_t)((int32_t)mx * mx) + (uint32_t)my * my;
                mag[y][x] = (uint16_t)(65535u - s);
            } else
                mag[y][x] = (uint16_t)img[y][x];  /* border: copy of input (undefined in SODA) */
        }
    }
}
}
