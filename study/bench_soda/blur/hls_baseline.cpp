/* Naive Vitis HLS baseline of plain.cpp for SODA kernel 'blur': same top/signature, m_axi per array + s_axilite,
 * no optimisation pragmas. NOTE: large static local buffer(s) (intermediate / ping-pong, up to 8 MB for 128^3 float)
 * are fine for csim but will map to BRAM/URAM (or fail to fit) in csynth; a real design must tile or stream. */
#include <math.h>
#include <stdint.h>
#include "blur.h"

extern "C" {
void workload(const uint16_t input[HEIGHT][WIDTH],
              uint16_t blur_y[HEIGHT][WIDTH])
{
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=blur_y offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=blur_y bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    static uint16_t bx[HEIGHT][WIDTH];  /* blur_x, SODA local (uint16) */
    for (int y = 0; y < HEIGHT - 2; y++) {
        for (int x = 0; x < WIDTH; x++) {
            bx[y][x] = (uint16_t)((input[y][x] + input[y+1][x] + input[y+2][x]) / 3);
        }
    }
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            if (x < WIDTH - 2 && y < HEIGHT - 2)
                blur_y[y][x] = (uint16_t)((bx[y][x] + bx[y][x+1] + bx[y][x+2]) / 3);
            else
                blur_y[y][x] = input[y][x];  /* border: copy of input (undefined in SODA) */
        }
    }
}
}
