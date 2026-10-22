/* plain C reference for SODA kernel 'denoise3d' (tests/src/denoise3d.soda), 3D TV denoise (6 face neighbours), valid [2,n-2)^3.
 * Layout: SODA name(dx,dy[,dz]) == name[y+dy][x+dx] ([z+dz][y+dy][x+dx]); first SODA index fastest.
 * Valid region (SODA defined): x,y,z in [2, size-2). Outside it SODA output is garbage; here the output border
 * is defined as a copy of the (first/main) input and the testbench never checks it.
 * Large static local arrays are used for intermediate/ping-pong buffers. */
#include <math.h>
#include <stdint.h>
#include "denoise3d.h"

extern "C" {
void workload(const float f[DEPTH][HEIGHT][WIDTH],
              const float u[DEPTH][HEIGHT][WIDTH],
              float output[DEPTH][HEIGHT][WIDTH])
{
    static float g[DEPTH][HEIGHT][WIDTH];  /* SODA local g, valid on [1,size-1)^3 */
    for (int z = 1; z < DEPTH - 1; z++) {
        for (int y = 1; y < HEIGHT - 1; y++) {
            for (int x = 1; x < WIDTH - 1; x++) {
                float du = u[z][y][x] - u[z][y-1][x];
                float dd = u[z][y][x] - u[z][y+1][x];
                float dl = u[z][y][x] - u[z][y][x-1];
                float dr = u[z][y][x] - u[z][y][x+1];
                float di = u[z][y][x] - u[z-1][y][x];
                float dO = u[z][y][x] - u[z+1][y][x];
                g[z][y][x] = 1.0f / sqrtf(0.00005f + du * du + dd * dd + dl * dl + dr * dr + di * di + dO * dO);
            }
        }
    }
    for (int z = 0; z < DEPTH; z++) {
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                if (x >= 2 && x < WIDTH - 2 && y >= 2 && y < HEIGHT - 2 && z >= 2 && z < DEPTH - 2) {
                    float r0 = u[z][y][x] * f[z][y][x] * (1.0f / 0.03f);
                    float r1 = (r0 * (2.38944f + r0 * (0.950037f + r0))) / (4.65314f + r0 * (2.57541f + r0 * (1.48937f + r0)));
                    output[z][y][x] = (u[z][y][x] + 5.0f * (u[z][y][x+1] * g[z][y][x+1] + u[z][y][x-1] * g[z][y][x-1] +
                        u[z][y+1][x] * g[z][y+1][x] + u[z][y-1][x] * g[z][y-1][x] +
                        u[z+1][y][x] * g[z+1][y][x] + u[z-1][y][x] * g[z-1][y][x] + (1.0f / 0.03f) * f[z][y][x] * r1)) /
                        (1.0f + 5.0f * (g[z][y][x+1] + g[z][y][x-1] + g[z][y+1][x] + g[z][y-1][x] + g[z+1][y][x] + g[z-1][y][x] + (1.0f / 0.03f)));
                } else
                    output[z][y][x] = u[z][y][x];  /* border: copy of u (undefined in SODA) */
            }
        }
    }
}
}
