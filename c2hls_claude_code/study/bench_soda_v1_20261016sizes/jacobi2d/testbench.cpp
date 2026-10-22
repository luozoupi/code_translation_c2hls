/*
 * Strict Vitis HLS C-simulation testbench for SODA kernel 'jacobi2d'.
 * Independent golden: plain.cpp's workload copied below as workload_ref (identical inputs, srand(42)).
 * Inputs (t1): read-only. Output t0: pure output, poisoned with 12345.0f before the call.
 * ONLY the valid interior x,y in [10, size-10) is compared (the whole of it); the border is undefined in SODA.
 * Tolerance: float tol 1e-3*fmax(1,|ref|), NaN is an error.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "jacobi2d.h"

/* ---- reference: copy of plain.cpp (static, renamed) ---- */
static void stage_ref(const float src[HEIGHT][WIDTH], float dst[HEIGHT][WIDTH])
{
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            if (x >= 1 && x < WIDTH - 1 && y >= 1 && y < HEIGHT - 1)
                dst[y][x] = (src[y+1][x] + src[y][x+1] + src[y][x] + src[y-1][x] + src[y][x-1]) * 0.2f;
            else
                dst[y][x] = src[y][x];  /* border: copy of src (undefined in SODA) */
        }
    }
}

static void workload_ref(const float t1[HEIGHT][WIDTH],
              float t0[HEIGHT][WIDTH])
{
    static float tmp[HEIGHT][WIDTH];  /* ping-pong buffer for chained iterations */
    stage_ref(t1, t0);  /* iteration 1 */
    for (int it = 1; it < ITERATE; it++) {
        stage_ref(t0, tmp);
        for (int y = 0; y < HEIGHT; y++)
            for (int x = 0; x < WIDTH; x++)
                t0[y][x] = tmp[y][x];
    }
}

static float t1[HEIGHT][WIDTH];
static float t0_dut[HEIGHT][WIDTH];
static float t0_ref[HEIGHT][WIDTH];

int main(void)
{
    srand(42);
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            t1[y][x] = (float)rand() / (float)RAND_MAX;
        }
    }
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            t0_dut[y][x] = 12345.0f;
        }
    }
    workload_ref(t1, t0_ref);
    workload(t1, t0_dut);
    int errors = 0;
    for (int y = 10; y < HEIGHT - 10; y++) {
        for (int x = 10; x < WIDTH - 10; x++) {
            float d = t0_dut[y][x];
            float r = t0_ref[y][x];
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
