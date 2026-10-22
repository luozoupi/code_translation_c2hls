/* Naive Vitis HLS baseline of plain.cpp for SODA kernel 'erosion': same top/signature, m_axi per array + s_axilite,
 * no optimisation pragmas. NOTE: large static local buffer(s) (intermediate / ping-pong, up to 8 MB for 128^3 float)
 * are fine for csim but will map to BRAM/URAM (or fail to fit) in csynth; a real design must tile or stream. */
#include <math.h>
#include <stdint.h>
#include "erosion.h"

extern "C" {
void workload(const int16_t input[HEIGHT][WIDTH],
              int16_t output[HEIGHT][WIDTH])
{
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    static int16_t tmp[HEIGHT][WIDTH];  /* SODA local int16 */
    for (int y = 9; y < HEIGHT - 9; y++) {
        for (int x = 0; x < WIDTH; x++) {
            int16_t m = input[y-9][x];
            for (int k = -9; k <= 9; k++) if (input[y+k][x] < m) m = input[y+k][x];
            tmp[y][x] = m;
        }
    }
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            if (x >= 9 && x < WIDTH - 9 && y >= 9 && y < HEIGHT - 9) {
                int16_t m = tmp[y][x-9];
                for (int k = -9; k <= 9; k++) if (tmp[y][x+k] < m) m = tmp[y][x+k];
                output[y][x] = m;
            } else
                output[y][x] = input[y][x];  /* border: copy of input (undefined in SODA) */
        }
    }
}
}
