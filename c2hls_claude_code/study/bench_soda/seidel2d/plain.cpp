/* plain C reference for SODA kernel 'seidel2d' (tests/src/seidel2d.soda), 3x3 box average * .1111111f (Jacobi-style, double-buffered), 9 chained applications.
 * Layout: SODA name(dx,dy[,dz]) == name[y+dy][x+dx] ([z+dz][y+dy][x+dx]); first SODA index fastest.
 * Valid region (SODA defined): x,y in [9, size-9). Outside it SODA output is garbage; here the output border
 * is defined as a copy of the (first/main) input and the testbench never checks it.
 * ITERATE=9 chained applications (SODA unrolls them into one dataflow); every stage writes its border as a copy of its
 * source, so the valid region shrinks by 1 per iteration: [9, size-9).
 * Large static local arrays are used for intermediate/ping-pong buffers. */
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
