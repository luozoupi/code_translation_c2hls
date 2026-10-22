/* plain C reference for SODA kernel 'erosion' (tests/src/erosion.soda), tmp[y][x]=min_{k=-9..9} in[y+k][x]; out[y][x]=min_{k=-9..9} tmp[y][x+k] (centred 19x19 min).
 * Layout: SODA name(dx,dy[,dz]) == name[y+dy][x+dx] ([z+dz][y+dy][x+dx]); first SODA index fastest.
 * Valid region (SODA defined): x,y in [9, size-9). Outside it SODA output is garbage; here the output border
 * is defined as a copy of the (first/main) input and the testbench never checks it.
 * Large static local arrays are used for intermediate/ping-pong buffers. */
#include <math.h>
#include <stdint.h>
#include "erosion.h"

extern "C" {
void workload(const int16_t input[HEIGHT][WIDTH],
              int16_t output[HEIGHT][WIDTH])
{
    static int16_t tmp[HEIGHT][WIDTH];  /* SODA local int16 */
    for (int y = 9; y < HEIGHT - 9; y++) {
        for (int x = 0; x < WIDTH; x++) {
            int16_t m = input[y-9][x];
            for (int k = -9; k <= 9; k++) if (input[y+k][x] < m) m = input[y+k][x];
            tmp[y][x] = m;
        }
    }
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            if (x >= 9 && x < WIDTH - 9 && y >= 9 && y < HEIGHT - 9) {
                int16_t m = tmp[y][x-9];
                for (int k = -9; k <= 9; k++) if (tmp[y][x+k] < m) m = tmp[y][x+k];
                output[y][x] = m;
            } else
                output[y][x] = input[y][x];  /* border: copy of input (undefined in SODA) */
        }
    }
}
}
