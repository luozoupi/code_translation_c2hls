/*
 * Strict Vitis HLS C-simulation testbench for SODA kernel 'heat3d'.
 * Independent golden: plain.cpp's workload copied below as workload_ref (identical inputs, srand(42)).
 * Inputs (in): read-only. Output out: pure output, poisoned with 12345.0f before the call.
 * ONLY the valid interior x,y,z in [5, size-5) is compared (the whole of it); the border is undefined in SODA.
 * Tolerance: float tol 1e-3*fmax(1,|ref|), NaN is an error.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "heat3d.h"

/* ---- reference: copy of plain.cpp (static, renamed) ---- */
static void stage_ref(const float src[DEPTH][HEIGHT][WIDTH], float dst[DEPTH][HEIGHT][WIDTH])
{
    for (int z = 0; z < DEPTH; z++) {
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                if (x >= 1 && x < WIDTH - 1 && y >= 1 && y < HEIGHT - 1 && z >= 1 && z < DEPTH - 1)
                    dst[z][y][x] = .125f * src[z][y][x+1] + .125f * src[z][y][x-1] + .125f * src[z][y+1][x] + .125f * src[z][y-1][x] + .125f * src[z+1][y][x] + .125f * src[z-1][y][x] + .25f * src[z][y][x];
                else
                    dst[z][y][x] = src[z][y][x];  /* border: copy of src (undefined in SODA) */
            }
        }
    }
}

static void workload_ref(const float in[DEPTH][HEIGHT][WIDTH],
              float out[DEPTH][HEIGHT][WIDTH])
{
    static float tmp[DEPTH][HEIGHT][WIDTH];  /* ping-pong buffer for chained iterations */
    stage_ref(in, out);  /* iteration 1 */
    for (int it = 1; it < ITERATE; it++) {
        stage_ref(out, tmp);
        for (int z = 0; z < DEPTH; z++)
            for (int y = 0; y < HEIGHT; y++)
                for (int x = 0; x < WIDTH; x++)
                    out[z][y][x] = tmp[z][y][x];
    }
}

static float in[DEPTH][HEIGHT][WIDTH];
static float out_dut[DEPTH][HEIGHT][WIDTH];
static float out_ref[DEPTH][HEIGHT][WIDTH];

int main(void)
{
    srand(42);
    for (int z = 0; z < DEPTH; z++) {
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                in[z][y][x] = (float)rand() / (float)RAND_MAX;
            }
        }
    }
    for (int z = 0; z < DEPTH; z++) {
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                out_dut[z][y][x] = 12345.0f;
            }
        }
    }
    workload_ref(in, out_ref);
    workload(in, out_dut);
    int errors = 0;
    for (int z = 5; z < DEPTH - 5; z++) {
        for (int y = 5; y < HEIGHT - 5; y++) {
            for (int x = 5; x < WIDTH - 5; x++) {
                float d = out_dut[z][y][x];
                float r = out_ref[z][y][x];
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
