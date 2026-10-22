/* Naive Vitis HLS baseline of plain.cpp for SODA kernel 'heat3d': same top/signature, m_axi per array + s_axilite,
 * no optimisation pragmas. NOTE: large static local buffer(s) (intermediate / ping-pong, up to 8 MB for 128^3 float)
 * are fine for csim but will map to BRAM/URAM (or fail to fit) in csynth; a real design must tile or stream. */
#include <math.h>
#include <stdint.h>
#include "heat3d.h"

static void stage(const float src[DEPTH][HEIGHT][WIDTH], float dst[DEPTH][HEIGHT][WIDTH])
{
    for (int z = 0; z < DEPTH; z++) {
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                if (x >= 1 && x < WIDTH - 1 && y >= 1 && y < HEIGHT - 1 && z >= 1 && z < DEPTH - 1)
                    dst[z][y][x] = .125f * src[z][y][x+1] + .125f * src[z][y][x-1] + .125f * src[z][y+1][x] + .125f * src[z][y-1][x] + .125f * src[z+1][y][x] + .125f * src[z-1][y][x] + .25f * src[z][y][x];
                else
                    dst[z][y][x] = src[z][y][x];  /* border: copy of src (undefined in SODA) */
            }
        }
    }
}

extern "C" {
void workload(const float in[DEPTH][HEIGHT][WIDTH],
              float out[DEPTH][HEIGHT][WIDTH])
{
#pragma HLS INTERFACE m_axi port=in offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=in bundle=control
#pragma HLS INTERFACE s_axilite port=out bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    static float tmp[DEPTH][HEIGHT][WIDTH];  /* ping-pong buffer for chained iterations */
    stage(in, out);  /* iteration 1 */
    for (int it = 1; it < ITERATE; it++) {
        stage(out, tmp);
        for (int z = 0; z < DEPTH; z++)
            for (int y = 0; y < HEIGHT; y++)
                for (int x = 0; x < WIDTH; x++)
                    out[z][y][x] = tmp[z][y][x];
    }
}
}
