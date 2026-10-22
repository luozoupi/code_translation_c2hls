/* plain C reference for SODA kernel 'xcorr' (tests/src/xcorr.soda), t1[y][x]=(int16)sum_{k=-9..9} in[y+k][x]; t2[y][x]=(int16)sum_{k=-9..9} t1[y][x+k]; out=(int16)(((int32)t2-in)*in/256) (int promotion, C truncating division).
 * Layout: SODA name(dx,dy[,dz]) == name[y+dy][x+dx] ([z+dz][y+dy][x+dx]); first SODA index fastest.
 * Valid region (SODA defined): x,y in [9, size-9). Outside it SODA output is garbage; here the output border
 * is defined as a copy of the (first/main) input and the testbench never checks it.
 * Large static local arrays are used for intermediate/ping-pong buffers. */
#include <math.h>
#include <stdint.h>
#include "xcorr.h"

extern "C" {
void workload(const int16_t input[HEIGHT][WIDTH],
              int16_t tmp3[HEIGHT][WIDTH])
{
    static int16_t tmp[HEIGHT][WIDTH];  /* SODA local int16 */
    for (int y = 9; y < HEIGHT - 9; y++) {
        for (int x = 0; x < WIDTH; x++) {
            int s = 0;
            for (int k = -9; k <= 9; k++) s += input[y+k][x];
            tmp[y][x] = (int16_t)s;  /* stored to int16: wraps (SODA local int16) */
        }
    }
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            if (x >= 9 && x < WIDTH - 9 && y >= 9 && y < HEIGHT - 9) {
                int s = 0;
                for (int k = -9; k <= 9; k++) s += tmp[y][x+k];
                int16_t t2 = (int16_t)s;  /* tmp2 local int16: wraps */
                tmp3[y][x] = (int16_t)(((int32_t)t2 - input[y][x]) * input[y][x] / 256);
            } else
                tmp3[y][x] = input[y][x];  /* border: copy of input (undefined in SODA) */
        }
    }
}
}
