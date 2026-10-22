/* Naive Vitis HLS baseline of plain.cpp for SODA kernel 'contrast': same top/signature, m_axi per array + s_axilite,
 * no optimisation pragmas. NOTE: large static local buffer(s) (intermediate / ping-pong, up to 8 MB for 128^3 float)
 * are fine for csim but will map to BRAM/URAM (or fail to fit) in csynth; a real design must tile or stream. */
#include <math.h>
#include <stdint.h>
#include "contrast.h"

extern "C" {
void workload(const float input[HEIGHT][WIDTH],
              float output[HEIGHT][WIDTH])
{
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    /* 17x17 weights, wt[dy][dx]; 197 non-zero taps copied from tests/src/contrast.soda; 0 = no tap */
    static const int wt[17][17] = {
        {0,0,0,0,0,0,0,0,-106,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,-90,-76,-67,-64,-67,-76,-90,0,0,0,0,0},
        {0,0,0,-98,-73,-52,-36,-26,-22,-26,-36,-52,-73,-98,0,0,0},
        {0,0,-98,-67,-39,-15,3,14,18,14,3,-15,-39,-67,-98,0,0},
        {0,0,-73,-39,-8,18,37,50,54,50,37,18,-8,-39,-73,0,0},
        {0,-90,-52,-15,18,46,67,80,84,80,67,46,18,-15,-52,-90,0},
        {0,-76,-36,3,37,67,89,103,108,103,89,67,37,3,-36,-76,0},
        {0,-67,-26,14,50,80,103,117,122,117,103,80,50,14,-26,-67,0},
        {-106,-64,-22,18,54,84,108,122,127,122,108,84,54,18,-22,-64,-106},
        {0,-67,-26,14,50,80,103,117,122,117,103,80,50,14,-26,-67,0},
        {0,-76,-36,3,37,67,89,103,108,103,89,67,37,3,-36,-76,0},
        {0,-90,-52,-15,18,46,67,80,84,80,67,46,18,-15,-52,-90,0},
        {0,0,-73,-39,-8,18,37,50,54,50,37,18,-8,-39,-73,0,0},
        {0,0,-98,-67,-39,-15,3,14,18,14,3,-15,-39,-67,-98,0,0},
        {0,0,0,-98,-73,-52,-36,-26,-22,-26,-36,-52,-73,-98,0,0,0},
        {0,0,0,0,0,-90,-76,-67,-64,-67,-76,-90,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,-106,0,0,0,0,0,0,0,0},
    };
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            if (x < WIDTH - 16 && y < HEIGHT - 16) {
                float acc = 0.0f;
                for (int dy = 0; dy < 17; dy++)
                    for (int dx = 0; dx < 17; dx++)
                        if (wt[dy][dx] != 0) acc += input[y+dy][x+dx] * (float)wt[dy][dx];
                output[y][x] = acc;
            } else
                output[y][x] = input[y][x];  /* border: copy of input (undefined in SODA) */
        }
    }
}
}
