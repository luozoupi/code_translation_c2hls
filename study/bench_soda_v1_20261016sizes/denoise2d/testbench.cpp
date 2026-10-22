/*
 * Strict Vitis HLS C-simulation testbench for SODA kernel 'denoise2d'.
 * Independent golden: plain.cpp's workload copied below as workload_ref (identical inputs, srand(42)).
 * Inputs (f, u): read-only. Output output: pure output, poisoned with 12345.0f before the call.
 * ONLY the valid interior x,y in [2, size-2) is compared (the whole of it); the border is undefined in SODA.
 * Tolerance: float tol 1e-3*fmax(1,|ref|), NaN is an error.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "denoise2d.h"

/* ---- reference: copy of plain.cpp (static, renamed) ---- */
static void workload_ref(const float f[HEIGHT][WIDTH],
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

static float f[HEIGHT][WIDTH];
static float u[HEIGHT][WIDTH];
static float output_dut[HEIGHT][WIDTH];
static float output_ref[HEIGHT][WIDTH];

int main(void)
{
    srand(42);
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            f[y][x] = (float)rand() / (float)RAND_MAX;
        }
    }
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            u[y][x] = (float)rand() / (float)RAND_MAX;
        }
    }
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            output_dut[y][x] = 12345.0f;
        }
    }
    workload_ref(f, u, output_ref);
    workload(f, u, output_dut);
    int errors = 0;
    for (int y = 2; y < HEIGHT - 2; y++) {
        for (int x = 2; x < WIDTH - 2; x++) {
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
