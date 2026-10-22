/* plain C reference for SODA kernel 'blur' (tests/src/blur.soda), blur_x(x,y)=(in[y][x]+in[y+1][x]+in[y+2][x])/3 ; out=(bx[y][x]+bx[y][x+1]+bx[y][x+2])/3, both stored uint16 (min-corner anchored).
 * Layout: SODA name(dx,dy[,dz]) == name[y+dy][x+dx] ([z+dz][y+dy][x+dx]); first SODA index fastest.
 * Valid region (SODA defined): x,y in [0, size-2). Outside it SODA output is garbage; here the output border
 * is defined as a copy of the (first/main) input and the testbench never checks it.
 * Large static local arrays are used for intermediate/ping-pong buffers. */
#include <math.h>
#include <stdint.h>
#include "blur.h"

extern "C" {
void workload(const uint16_t input[HEIGHT][WIDTH],
              uint16_t blur_y[HEIGHT][WIDTH])
{
    static uint16_t bx[HEIGHT][WIDTH];  /* blur_x, SODA local (uint16) */
    for (int y = 0; y < HEIGHT - 2; y++) {
        for (int x = 0; x < WIDTH; x++) {
            bx[y][x] = (uint16_t)((input[y][x] + input[y+1][x] + input[y+2][x]) / 3);
        }
    }
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            if (x < WIDTH - 2 && y < HEIGHT - 2)
                blur_y[y][x] = (uint16_t)((bx[y][x] + bx[y][x+1] + bx[y][x+2]) / 3);
            else
                blur_y[y][x] = input[y][x];  /* border: copy of input (undefined in SODA) */
        }
    }
}
}
