/* plain C reference for SODA kernel 'jacobi3d' (tests/src/jacobi3d.soda), t0=(c+x+ +x- +y+ +y- +z+ +z-)*0.142857142f, 6 chained applications.
 * Layout: SODA name(dx,dy[,dz]) == name[y+dy][x+dx] ([z+dz][y+dy][x+dx]); first SODA index fastest.
 * Valid region (SODA defined): x,y,z in [6, size-6). Outside it SODA output is garbage; here the output border
 * is defined as a copy of the (first/main) input and the testbench never checks it.
 * ITERATE=6 chained applications (SODA unrolls them into one dataflow); every stage writes its border as a copy of its
 * source, so the valid region shrinks by 1 per iteration: [6, size-6).
 * Large static local arrays are used for intermediate/ping-pong buffers. */
#include <math.h>
#include <stdint.h>
#include "jacobi3d.h"

static void stage(const float src[DEPTH][HEIGHT][WIDTH], float dst[DEPTH][HEIGHT][WIDTH])
{
    for (int z = 0; z < DEPTH; z++) {
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                if (x >= 1 && x < WIDTH - 1 && y >= 1 && y < HEIGHT - 1 && z >= 1 && z < DEPTH - 1)
                    dst[z][y][x] = (src[z][y][x] + src[z][y][x+1] + src[z][y][x-1] + src[z][y+1][x] + src[z][y-1][x] + src[z+1][y][x] + src[z-1][y][x]) * 0.142857142f;
                else
                    dst[z][y][x] = src[z][y][x];  /* border: copy of src (undefined in SODA) */
            }
        }
    }
}

extern "C" {
void workload(const float t1[DEPTH][HEIGHT][WIDTH],
              float t0[DEPTH][HEIGHT][WIDTH])
{
    static float tmp[DEPTH][HEIGHT][WIDTH];  /* ping-pong buffer for chained iterations */
    stage(t1, t0);  /* iteration 1 */
    for (int it = 1; it < ITERATE; it++) {
        stage(t0, tmp);
        for (int z = 0; z < DEPTH; z++)
            for (int y = 0; y < HEIGHT; y++)
                for (int x = 0; x < WIDTH; x++)
                    t0[z][y][x] = tmp[z][y][x];
    }
}
}
