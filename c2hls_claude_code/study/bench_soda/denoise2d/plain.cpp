/* plain C reference for SODA kernel 'denoise2d' (tests/src/denoise2d.soda), TV denoise: g=1/sqrt(1+sum diff^2) on [1,n-1), output uses g at 4 neighbours -> valid [2,n-2).
 * Layout: SODA name(dx,dy[,dz]) == name[y+dy][x+dx] ([z+dz][y+dy][x+dx]); first SODA index fastest.
 * Valid region (SODA defined): x,y in [2, size-2). Outside it SODA output is garbage; here the output border
 * is defined as a copy of the (first/main) input and the testbench never checks it.
 * Large static local arrays are used for intermediate/ping-pong buffers. */
#include <math.h>
#include <stdint.h>
#include "denoise2d.h"

extern "C" {
void workload(const float f[HEIGHT][WIDTH],
              const float u[HEIGHT][WIDTH],
              float output[HEIGHT][WIDTH])
{
    static float g[HEIGHT][WIDTH];  /* SODA local g, valid on [1,size-1) */
    for (int y = 1; y < HEIGHT - 1; y++) {
        for (int x = 1; x < WIDTH - 1; x++) {
            float du = u[y][x] - u[y-1][x];
            float dd = u[y][x] - u[y+1][x];
            float dl = u[y][x] - u[y][x-1];
            float dr = u[y][x] - u[y][x+1];
            g[y][x] = 1.0f / sqrtf(1.0f + du * du + dd * dd + dl * dl + dr * dr);
        }
    }
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            if (x >= 2 && x < WIDTH - 2 && y >= 2 && y < HEIGHT - 2) {
                float r0 = u[y][x] * f[y][x] * 4.9f;
                float r1 = (r0 * (2.5f + r0 * (10.2f + r0))) * (4.3f + r0 * (5.4f + r0 * (6.3f + r0)));
                output[y][x] = (u[y][x] + 7.7f *
                    (u[y+1][x] * g[y+1][x] + u[y-1][x] * g[y-1][x] + u[y][x-1] * g[y][x-1] + u[y][x+1] * g[y][x+1] +
                     5.7f * f[y][x] * r1)) * (11.1f + 7.7f *
                    (g[y+1][x] + g[y-1][x] + g[y][x-1] + g[y][x+1] + 5.7f));
            } else
                output[y][x] = u[y][x];  /* border: copy of u (undefined in SODA) */
        }
    }
}
}
