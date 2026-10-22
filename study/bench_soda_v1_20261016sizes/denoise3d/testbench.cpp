/*
 * Strict Vitis HLS C-simulation testbench for SODA kernel 'denoise3d'.
 * Independent golden: plain.cpp's workload copied below as workload_ref (identical inputs, srand(42)).
 * Inputs (f, u): read-only. Output output: pure output, poisoned with 12345.0f before the call.
 * ONLY the valid interior x,y,z in [2, size-2) is compared (the whole of it); the border is undefined in SODA.
 * Tolerance: float tol 1e-3*fmax(1,|ref|), NaN is an error.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "denoise3d.h"

/* ---- reference: copy of plain.cpp (static, renamed) ---- */
static void workload_ref(const float f[DEPTH][HEIGHT][WIDTH],
              const float u[DEPTH][HEIGHT][WIDTH],
              float output[DEPTH][HEIGHT][WIDTH])
{
    static float g[DEPTH][HEIGHT][WIDTH];  /* SODA local g, valid on [1,size-1)^3 */
    for (int z = 1; z < DEPTH - 1; z++) {
        for (int y = 1; y < HEIGHT - 1; y++) {
            for (int x = 1; x < WIDTH - 1; x++) {
                float du = u[z][y][x] - u[z][y-1][x];
                float dd = u[z][y][x] - u[z][y+1][x];
                float dl = u[z][y][x] - u[z][y][x-1];
                float dr = u[z][y][x] - u[z][y][x+1];
                float di = u[z][y][x] - u[z-1][y][x];
                float dO = u[z][y][x] - u[z+1][y][x];
                g[z][y][x] = 1.0f / sqrtf(0.00005f + du * du + dd * dd + dl * dl + dr * dr + di * di + dO * dO);
            }
        }
    }
    for (int z = 0; z < DEPTH; z++) {
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                if (x >= 2 && x < WIDTH - 2 && y >= 2 && y < HEIGHT - 2 && z >= 2 && z < DEPTH - 2) {
                    float r0 = u[z][y][x] * f[z][y][x] * (1.0f / 0.03f);
                    float r1 = (r0 * (2.38944f + r0 * (0.950037f + r0))) / (4.65314f + r0 * (2.57541f + r0 * (1.48937f + r0)));
                    output[z][y][x] = (u[z][y][x] + 5.0f * (u[z][y][x+1] * g[z][y][x+1] + u[z][y][x-1] * g[z][y][x-1] +
                        u[z][y+1][x] * g[z][y+1][x] + u[z][y-1][x] * g[z][y-1][x] +
                        u[z+1][y][x] * g[z+1][y][x] + u[z-1][y][x] * g[z-1][y][x] + (1.0f / 0.03f) * f[z][y][x] * r1)) /
                        (1.0f + 5.0f * (g[z][y][x+1] + g[z][y][x-1] + g[z][y+1][x] + g[z][y-1][x] + g[z+1][y][x] + g[z-1][y][x] + (1.0f / 0.03f)));
                } else
                    output[z][y][x] = u[z][y][x];  /* border: copy of u (undefined in SODA) */
            }
        }
    }
}

static float f[DEPTH][HEIGHT][WIDTH];
static float u[DEPTH][HEIGHT][WIDTH];
static float output_dut[DEPTH][HEIGHT][WIDTH];
static float output_ref[DEPTH][HEIGHT][WIDTH];

int main(void)
{
    srand(42);
    for (int z = 0; z < DEPTH; z++) {
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                f[z][y][x] = (float)rand() / (float)RAND_MAX;
            }
        }
    }
    for (int z = 0; z < DEPTH; z++) {
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                u[z][y][x] = (float)rand() / (float)RAND_MAX;
            }
        }
    }
    for (int z = 0; z < DEPTH; z++) {
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                output_dut[z][y][x] = 12345.0f;
            }
        }
    }
    workload_ref(f, u, output_ref);
    workload(f, u, output_dut);
    int errors = 0;
    for (int z = 2; z < DEPTH - 2; z++) {
        for (int y = 2; y < HEIGHT - 2; y++) {
            for (int x = 2; x < WIDTH - 2; x++) {
                float d = output_dut[z][y][x];
                float r = output_ref[z][y][x];
                int bad = (d != d) || !(fabs((double)d - (double)r) <= 1e-3 * fmax(1.0, fabs((double)r)));
                if (bad) {
                    if (errors < 10) printf("mismatch at x=%d y=%d z=%d: got %g expected %g\n", x, y, z, (double)d, (double)r);
                    errors++;
                }
            }
        }
    }
    if (errors) { printf("Failed with %d errors!\n", errors); return 1; }
    printf("Passed!\n");
    return 0;
}
