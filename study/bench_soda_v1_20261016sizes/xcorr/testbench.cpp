/*
 * Strict Vitis HLS C-simulation testbench for SODA kernel 'xcorr'.
 * Independent golden: plain.cpp's workload copied below as workload_ref (identical inputs, srand(42)).
 * Inputs (input): read-only. Output tmp3: pure output, poisoned with (int16_t)0x5A5A before the call.
 * ONLY the valid interior x,y in [9, size-9) is compared (the whole of it); the border is undefined in SODA.
 * Tolerance: integers exact.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "xcorr.h"

/* ---- reference: copy of plain.cpp (static, renamed) ---- */
static void workload_ref(const int16_t input[HEIGHT][WIDTH],
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

static int16_t input[HEIGHT][WIDTH];
static int16_t tmp3_dut[HEIGHT][WIDTH];
static int16_t tmp3_ref[HEIGHT][WIDTH];

int main(void)
{
    srand(42);
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            input[y][x] = (int16_t)(rand() & 0xFFFF);
        }
    }
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            tmp3_dut[y][x] = (int16_t)0x5A5A;
        }
    }
    workload_ref(input, tmp3_ref);
    workload(input, tmp3_dut);
    int errors = 0;
    for (int y = 9; y < HEIGHT - 9; y++) {
        for (int x = 9; x < WIDTH - 9; x++) {
            int16_t d = tmp3_dut[y][x];
            int16_t r = tmp3_ref[y][x];
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
