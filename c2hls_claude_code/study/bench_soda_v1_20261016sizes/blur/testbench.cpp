/*
 * Strict Vitis HLS C-simulation testbench for SODA kernel 'blur'.
 * Independent golden: plain.cpp's workload copied below as workload_ref (identical inputs, srand(42)).
 * Inputs (input): read-only. Output blur_y: pure output, poisoned with (uint16_t)0x5A5A before the call.
 * ONLY the valid interior x,y in [0, size-2) is compared (the whole of it); the border is undefined in SODA.
 * Tolerance: integers exact.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "blur.h"

/* ---- reference: copy of plain.cpp (static, renamed) ---- */
static void workload_ref(const uint16_t input[HEIGHT][WIDTH],
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

static uint16_t input[HEIGHT][WIDTH];
static uint16_t blur_y_dut[HEIGHT][WIDTH];
static uint16_t blur_y_ref[HEIGHT][WIDTH];

int main(void)
{
    srand(42);
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            input[y][x] = (uint16_t)(rand() & 0xFFFF);
        }
    }
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            blur_y_dut[y][x] = (uint16_t)0x5A5A;
        }
    }
    workload_ref(input, blur_y_ref);
    workload(input, blur_y_dut);
    int errors = 0;
    for (int y = 0; y < HEIGHT - 2; y++) {
        for (int x = 0; x < WIDTH - 2; x++) {
            uint16_t d = blur_y_dut[y][x];
            uint16_t r = blur_y_ref[y][x];
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
