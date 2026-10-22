/* Naive Vitis HLS baseline of plain.cpp for SODA kernel 'xcorr': same top/signature, m_axi per array + s_axilite,
 * no optimisation pragmas. NOTE: large static local buffer(s) (intermediate / ping-pong, up to 8 MB for 128^3 float)
 * are fine for csim but will map to BRAM/URAM (or fail to fit) in csynth; a real design must tile or stream. */
#include <math.h>
#include <stdint.h>
#include "xcorr.h"

extern "C" {
void workload(const int16_t input[HEIGHT][WIDTH],
              int16_t tmp3[HEIGHT][WIDTH])
{
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=tmp3 offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=tmp3 bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    static int16_t tmp[HEIGHT][WIDTH];  /* SODA local int16 */
    for (int y = 9; y < HEIGHT - 9; y++) {
        for (int x = 0; x < WIDTH; x++) {
            int s = 0;
            for (int k = -9; k <= 9; k++) s += input[y+k][x];
            tmp[y][x] = (int16_t)s;  /* stored to int16: wraps (SODA local int16) */
        }
    }
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            if (x >= 9 && x < WIDTH - 9 && y >= 9 && y < HEIGHT - 9) {
                int s = 0;
                for (int k = -9; k <= 9; k++) s += tmp[y][x+k];
                int16_t t2 = (int16_t)s;  /* tmp2 local int16: wraps */
                tmp3[y][x] = (int16_t)(((int32_t)t2 - input[y][x]) * input[y][x] / 256);
            } else
                tmp3[y][x] = input[y][x];  /* border: copy of input (undefined in SODA) */
        }
    }
}
}
