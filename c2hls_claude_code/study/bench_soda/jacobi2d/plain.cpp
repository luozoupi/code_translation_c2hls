/* plain C reference for SODA kernel 'jacobi2d' (tests/src/jacobi2d.soda), t0=(t1(0,1)+t1(1,0)+t1(0,0)+t1(0,-1)+t1(-1,0))*0.2f, 10 chained applications.
 * Layout: SODA name(dx,dy[,dz]) == name[y+dy][x+dx] ([z+dz][y+dy][x+dx]); first SODA index fastest.
 * Valid region (SODA defined): x,y in [10, size-10). Outside it SODA output is garbage; here the output border
 * is defined as a copy of the (first/main) input and the testbench never checks it.
 * ITERATE=10 chained applications (SODA unrolls them into one dataflow); every stage writes its border as a copy of its
 * source, so the valid region shrinks by 1 per iteration: [10, size-10).
 * Large static local arrays are used for intermediate/ping-pong buffers. */
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
