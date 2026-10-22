/*
 * Strict Vitis HLS C-simulation testbench for SODA kernel 'sobel2d'.
 * Independent golden: plain.cpp's workload copied below as workload_ref (identical inputs, srand(42)).
 * Inputs (img): read-only. Output mag: pure output, poisoned with (uint16_t)0x5A5A before the call.
 * ONLY the valid interior x,y in [1, size-1) is compared (the whole of it); the border is undefined in SODA.
 * Tolerance: integers exact.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "sobel2d.h"

/* ---- reference: copy of plain.cpp (static, renamed) ---- */
static void workload_ref(const int16_t img[HEIGHT][WIDTH],
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

static int16_t img[HEIGHT][WIDTH];
static uint16_t mag_dut[HEIGHT][WIDTH];
static uint16_t mag_ref[HEIGHT][WIDTH];

int main(void)
{
    srand(42);
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            img[y][x] = (int16_t)(rand() & 0xFFFF);
        }
    }
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            mag_dut[y][x] = (uint16_t)0x5A5A;
        }
    }
    workload_ref(img, mag_ref);
    workload(img, mag_dut);
    int errors = 0;
    for (int y = 1; y < HEIGHT - 1; y++) {
        for (int x = 1; x < WIDTH - 1; x++) {
            uint16_t d = mag_dut[y][x];
            uint16_t r = mag_ref[y][x];
            int bad = (d != r);
            if (bad) {
                if (errors < 10) printf("mismatch at x=%d y=%d: got %ld expected %ld\n", x, y, (long)d, (long)r);
                errors++;
            }
        }
    }
    if (errors) { printf("Failed with %d errors!\n", errors); return 1; }
    printf("Passed!\n");
    return 0;
}
