/*
 * Strict Vitis HLS C-simulation testbench for lc_dilate (no testbench shipped originally).
 * Golden: plain.cpp's workload() copied below as workload_ref() and run on identical copies of the deterministic input
 * (srand(42), img values 0..1). workload(result, img): result[GRID_ROWS*GRID_COLS] is a pure output (every element
 * written) -> poisoned with 12345.0f; img is input only. Every element checked, float tol 1e-3*fmax(1,|ref|).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "dilate.h"

extern "C" void workload(float result[GRID_ROWS * GRID_COLS], float img[(GRID_ROWS+2*MAX_RADIUS)*GRID_COLS]);

/* ---- reference: copy of plain.cpp ---- */



static void workload_ref(float result[GRID_ROWS * GRID_COLS], float img[(GRID_ROWS+2*MAX_RADIUS)*GRID_COLS])
{


	int starting_idx = MAX_RADIUS*GRID_COLS;

	bool strel[25] = { 0, 0, 1, 0, 0,
					   0, 1, 1, 1, 0,
					   1, 1, 1, 1, 1,
					   0, 1, 1, 1, 0,
					   0, 0, 1, 0, 0 };

	int radius_p = STREL_ROWS / 2;
	int radius_q = STREL_COLS / 2;

	for (int i = 0; i < GRID_ROWS; i++) {
		for (int j = 0; j < GRID_COLS; j++) {
			float max = 0.0, temp;
			for (int m = 0; m < STREL_ROWS; m++) {
				for (int n = 0; n < STREL_COLS; n++) {
					int p = i - radius_p + m;
					int q = j - radius_q + n;
					if (p >= 0 && q >= 0 && p < GRID_ROWS && q < GRID_COLS && strel[m * STREL_COLS + n] != 0) {
						temp = img[starting_idx + p * GRID_COLS + q];
						if (temp > max) max = temp;
					}
				}
			}
			result[i * GRID_COLS + j] = max;
		}
	}

	return;
}

/* ---- end reference ---- */

static int errors = 0;
static void chk_f(const char *name, int i, float d, float r) {
    if (!isfinite(r) || !(fabsf(d - r) <= 1e-3f * fmaxf(1.0f, fabsf(r)))) {   /* NaN/inf in DUT output (or a bad reference) is an error */
        if (errors < 10) printf("%s[%d]: got %g expected %g\n", name, i, (double)d, (double)r);
        errors++;
    }
}
static float frand(float lo, float hi) { return lo + (hi - lo) * ((float)rand() / (float)RAND_MAX); }

static float img[(GRID_ROWS+2*MAX_RADIUS)*GRID_COLS], res_ref[GRID_ROWS*GRID_COLS], res_dut[GRID_ROWS*GRID_COLS];

int main() {
    srand(42);
    for (int i = 0; i < (GRID_ROWS+2*MAX_RADIUS)*GRID_COLS; i++) img[i] = frand(0.0f, 1.0f);
    for (int i = 0; i < GRID_ROWS*GRID_COLS; i++) { res_ref[i] = 12345.0f; res_dut[i] = 12345.0f; }
    workload_ref(res_ref, img);
    workload(res_dut, img);
    for (int i = 0; i < GRID_ROWS*GRID_COLS; i++) chk_f("result", i, res_dut[i], res_ref[i]);
    if (errors) { printf("Failed with %d errors!\n", errors); return 1; }
    printf("Passed!\n");
    return 0;
}
