/* plain C reference for SODA kernel 'sobel2d' (tests/src/sobel2d.soda), mx=(int16)(sobel x), my=(uint16)(sobel y), mag=(uint16)(65535-(mx*mx+my*my)).
 * Layout: SODA name(dx,dy[,dz]) == name[y+dy][x+dx] ([z+dz][y+dy][x+dx]); first SODA index fastest.
 * Valid region (SODA defined): x,y in [1, size-1). Outside it SODA output is garbage; here the output border
 * is defined as a copy of the (first/main) input and the testbench never checks it.
 * Large static local arrays are used for intermediate/ping-pong buffers. */
#include <math.h>
#include <stdint.h>
#include "sobel2d.h"

extern "C" {
void workload(const int16_t img[HEIGHT][WIDTH],
              uint16_t mag[HEIGHT][WIDTH])
{
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            if (x >= 1 && x < WIDTH - 1 && y >= 1 && y < HEIGHT - 1) {
                /* mag_x: int16 local; mag_y: uint16 local (truncate at the store, as SODA does) */
                int16_t mx = (int16_t)((img[y-1][x+1] - img[y-1][x-1]) + (img[y][x+1] - img[y][x-1]) * 3 + (img[y+1][x+1] - img[y+1][x-1]));
                uint16_t my = (uint16_t)((img[y+1][x-1] - img[y-1][x-1]) + (img[y+1][x] - img[y-1][x]) * 3 + (img[y+1][x+1] - img[y-1][x+1]));
                /* SODA: 65535 - (mx*mx + my*my) in int, stored uint16. Done in uint32 here (no signed overflow UB);
                   the low 16 bits are identical. */
                uint32_t s = (uint32_t)((int32_t)mx * mx) + (uint32_t)my * my;
                mag[y][x] = (uint16_t)(65535u - s);
            } else
                mag[y][x] = (uint16_t)img[y][x];  /* border: copy of input (undefined in SODA) */
        }
    }
}
}
