/* Naive Vitis HLS baseline of plain.cpp for SODA kernel 'jacobi2d': same top/signature, m_axi per array + s_axilite,
 * no optimisation pragmas. NOTE: large static local buffer(s) (intermediate / ping-pong, up to 8 MB for 128^3 float)
 * are fine for csim but will map to BRAM/URAM (or fail to fit) in csynth; a real design must tile or stream. */
#include <math.h>
#include <stdint.h>
#include "jacobi2d.h"

static void stage(const float src[HEIGHT][WIDTH], float dst[HEIGHT][WIDTH])
{
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            if (x >= 1 && x < WIDTH - 1 && y >= 1 && y < HEIGHT - 1)
                dst[y][x] = (src[y+1][x] + src[y][x+1] + src[y][x] + src[y-1][x] + src[y][x-1]) * 0.2f;
            else
                dst[y][x] = src[y][x];  /* border: copy of src (undefined in SODA) */
        }
    }
}

extern "C" {
void workload(const float t1[HEIGHT][WIDTH],
              float t0[HEIGHT][WIDTH])
{
#pragma HLS INTERFACE m_axi port=t1 offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=t0 offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=t1 bundle=control
#pragma HLS INTERFACE s_axilite port=t0 bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    static float tmp[HEIGHT][WIDTH];  /* ping-pong buffer for chained iterations */
    stage(t1, t0);  /* iteration 1 */
    for (int it = 1; it < ITERATE; it++) {
        stage(t0, tmp);
        for (int y = 0; y < HEIGHT; y++)
            for (int x = 0; x < WIDTH; x++)
                t0[y][x] = tmp[y][x];
    }
}
}
