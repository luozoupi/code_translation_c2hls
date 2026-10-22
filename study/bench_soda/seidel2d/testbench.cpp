/*
 * Strict Vitis HLS C-simulation testbench for SODA kernel 'seidel2d'.
 * Independent golden: plain.cpp's workload copied below as workload_ref (identical inputs, srand(42)).
 * Inputs (input): read-only. Output output: pure output, poisoned with 12345.0f before the call.
 * ONLY the valid interior x,y in [9, size-9) is compared (the whole of it); the border is undefined in SODA.
 * Tolerance: float tol 1e-3*fmax(1,|ref|), NaN is an error.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "seidel2d.h"

/* ---- reference: copy of plain.cpp (static, renamed) ---- */
static void stage_ref(const float src[HEIGHT][WIDTH], float dst[HEIGHT][WIDTH])
{
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            if (x >= 1 && x < WIDTH - 1 && y >= 1 && y < HEIGHT - 1)
                dst[y][x] = (src[y-1][x-1] + src[y][x-1] + src[y+1][x-1] + src[y-1][x] + src[y][x] + src[y+1][x] + src[y-1][x+1] + src[y][x+1] + src[y+1][x+1]) * .1111111f;
            else
                dst[y][x] = src[y][x];  /* border: copy of src (undefined in SODA) */
        }
    }
}

static void workload_ref(const float input[HEIGHT][WIDTH],
              float output[HEIGHT][WIDTH])
{
    static float tmp[HEIGHT][WIDTH];  /* ping-pong buffer for chained iterations */
    stage_ref(input, output);  /* iteration 1 */
    for (int it = 1; it < ITERATE; it++) {
        stage_ref(output, tmp);
        for (int y = 0; y < HEIGHT; y++)
            for (int x = 0; x < WIDTH; x++)
                output[y][x] = tmp[y][x];
    }
}

static float input[HEIGHT][WIDTH];
static float output_dut[HEIGHT][WIDTH];
static float output_ref[HEIGHT][WIDTH];

int main(void)
{
    srand(42);
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            input[y][x] = (float)rand() / (float)RAND_MAX;
        }
    }
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            output_dut[y][x] = 12345.0f;
        }
    }
    workload_ref(input, output_ref);
    workload(input, output_dut);
    int errors = 0;
    for (int y = 9; y < HEIGHT - 9; y++) {
        for (int x = 9; x < WIDTH - 9; x++) {
            float d = output_dut[y][x];
            float r = output_ref[y][x];
            int bad = (d != d) || !(fabs((double)d - (double)r) <= 1e-3 * fmax(1.0, fabs((double)r)));
            if (bad) {
                if (errors < 10) printf("mismatch at x=%d y=%d: got %g expected %g\n", x, y, (double)d, (double)r);
                errors++;
            }
        }
    }
    if (errors) { printf("Failed with %d errors!\n", errors); return 1; }
    printf("Passed!\n");
    return 0;
}
