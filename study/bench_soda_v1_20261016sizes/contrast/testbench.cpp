/*
 * Strict Vitis HLS C-simulation testbench for SODA kernel 'contrast'.
 * Independent golden: plain.cpp's workload copied below as workload_ref (identical inputs, srand(42)).
 * Inputs (input): read-only. Output output: pure output, poisoned with 12345.0f before the call.
 * ONLY the valid interior x,y in [0, size-16) is compared (the whole of it); the border is undefined in SODA.
 * Tolerance: float tol 1e-3*fmax(1,|ref|), NaN is an error.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "contrast.h"

/* ---- reference: copy of plain.cpp (static, renamed) ---- */
static void workload_ref(const float input[HEIGHT][WIDTH],
              float output[HEIGHT][WIDTH])
{
    /* 17x17 weights, wt[dy][dx]; 197 non-zero taps copied from tests/src/contrast.soda; 0 = no tap */
    static const int wt[17][17] = {
        {0,0,0,0,0,0,0,0,-106,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,-90,-76,-67,-64,-67,-76,-90,0,0,0,0,0},
        {0,0,0,-98,-73,-52,-36,-26,-22,-26,-36,-52,-73,-98,0,0,0},
        {0,0,-98,-67,-39,-15,3,14,18,14,3,-15,-39,-67,-98,0,0},
        {0,0,-73,-39,-8,18,37,50,54,50,37,18,-8,-39,-73,0,0},
        {0,-90,-52,-15,18,46,67,80,84,80,67,46,18,-15,-52,-90,0},
        {0,-76,-36,3,37,67,89,103,108,103,89,67,37,3,-36,-76,0},
        {0,-67,-26,14,50,80,103,117,122,117,103,80,50,14,-26,-67,0},
        {-106,-64,-22,18,54,84,108,122,127,122,108,84,54,18,-22,-64,-106},
        {0,-67,-26,14,50,80,103,117,122,117,103,80,50,14,-26,-67,0},
        {0,-76,-36,3,37,67,89,103,108,103,89,67,37,3,-36,-76,0},
        {0,-90,-52,-15,18,46,67,80,84,80,67,46,18,-15,-52,-90,0},
        {0,0,-73,-39,-8,18,37,50,54,50,37,18,-8,-39,-73,0,0},
        {0,0,-98,-67,-39,-15,3,14,18,14,3,-15,-39,-67,-98,0,0},
        {0,0,0,-98,-73,-52,-36,-26,-22,-26,-36,-52,-73,-98,0,0,0},
        {0,0,0,0,0,-90,-76,-67,-64,-67,-76,-90,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,-106,0,0,0,0,0,0,0,0},
    };
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            if (x < WIDTH - 16 && y < HEIGHT - 16) {
                float acc = 0.0f;
                for (int dy = 0; dy < 17; dy++)
                    for (int dx = 0; dx < 17; dx++)
                        if (wt[dy][dx] != 0) acc += input[y+dy][x+dx] * (float)wt[dy][dx];
                output[y][x] = acc;
            } else
                output[y][x] = input[y][x];  /* border: copy of input (undefined in SODA) */
        }
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
            input[y][x] = (float)(rand() % 256) / 16.0f;
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
    for (int y = 0; y < HEIGHT - 16; y++) {
        for (int x = 0; x < WIDTH - 16; x++) {
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
