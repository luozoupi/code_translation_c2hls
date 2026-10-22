/* Naive Vitis HLS baseline of plain.cpp for SODA kernel 'seidel2d': same top/signature, m_axi per array + s_axilite,
 * no optimisation pragmas. NOTE: large static local buffer(s) (intermediate / ping-pong, up to 8 MB for 128^3 float)
 * are fine for csim but will map to BRAM/URAM (or fail to fit) in csynth; a real design must tile or stream. */
#include <math.h>
#include <stdint.h>
#include "seidel2d.h"

static void stage(const float src[HEIGHT][WIDTH], float dst[HEIGHT][WIDTH])
{
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            if (x >= 1 && x < WIDTH - 1 && y >= 1 && y < HEIGHT - 1)
                dst[y][x] = (src[y-1][x-1] + src[y][x-1] + src[y+1][x-1] + src[y-1][x] + src[y][x] + src[y+1][x] + src[y-1][x+1] + src[y][x+1] + src[y+1][x+1]) * .1111111f;
            else
                dst[y][x] = src[y][x];  /* border: copy of src (undefined in SODA) */
        }
    }
}

extern "C" {
void workload(const float input[HEIGHT][WIDTH],
              float output[HEIGHT][WIDTH])
{
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    static float tmp[HEIGHT][WIDTH];  /* ping-pong buffer for chained iterations */
    stage(input, output);  /* iteration 1 */
    for (int it = 1; it < ITERATE; it++) {
        stage(output, tmp);
        for (int y = 0; y < HEIGHT; y++)
            for (int x = 0; x < WIDTH; x++)
                output[y][x] = tmp[y][x];
    }
}
}
