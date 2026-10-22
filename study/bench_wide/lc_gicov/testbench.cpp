/*
 * Strict Vitis HLS C-simulation testbench for lc_gicov (no testbench shipped originally).
 * Golden: plain.cpp's workload() copied below as workload_ref() and run on identical copies of the deterministic input
 * (srand(42), gradients in -1..1). workload(result, grad_x, grad_y): result is a pure output, poisoned with 12345.0f;
 * grad_x/grad_y are inputs. plain.cpp writes result only for rows/cols MAX_RADIUS..N-MAX_RADIUS-1 (the border is never
 * written, i.e. undefined), so exactly that interior region is checked: every element of it, float tol
 * 1e-3*fmax(1,|ref|), NaN = error.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "lc_gicov.h"

extern "C" void workload(float result[GRID_ROWS * GRID_COLS], float grad_x[GRID_ROWS * GRID_COLS], float grad_y[GRID_ROWS * GRID_COLS]);

/* ---- reference: copy of plain.cpp ---- */



static void workload_ref(float result[GRID_ROWS * GRID_COLS], float grad_x[GRID_ROWS * GRID_COLS], float grad_y[GRID_ROWS * GRID_COLS])
{

    
    

    float sin_angle[16] = {0, 0.382683432365090, 0.707106781186548, 0.923879532511287, 1, 0.923879532511287, 0.707106781186548, 0.382683432365090, 1.22464679914735e-16, -0.382683432365090, -0.707106781186548, -0.923879532511287, -1, -0.923879532511287, -0.707106781186548, -0.382683432365090};

    float cos_angle[16] = {1, 0.923879532511287, 0.707106781186548, 0.382683432365090, 6.12323399573677e-17, -0.382683432365090, -0.707106781186548, -0.923879532511287, -1, -0.923879532511287, -0.707106781186548, -0.382683432365090, -1.83697019872103e-16, 0.382683432365090, 0.707106781186547, 0.923879532511287};

    int tX[2][16] = {{1, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0}, {2, 1, 1, 0, 0, -1, -2, -2, -2, -2, -2, -1, 0, 0, 1, 1}};

    int tY[2][16] = {{0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1}, {0, 0, 1, 1, 1, 1, 1, 0, 0, -1, -2, -2, -2, -2, -2, -1}};

    int i, n, k;

	// Allocate memory for the result matrix
	// int GRID_ROWS = GRID_ROWS, GRID_COLS = GRID_COLS;
	float *gicov = (float*)malloc(GRID_ROWS * GRID_COLS * sizeof(float));


	// Scan from left to right, top to bottom, computing GICOV values
	for (i = MAX_RADIUS; i < GRID_COLS - MAX_RADIUS; i++) {
		float Grad[NPOINTS];
		int j, k, n, x, y;

		for (j = MAX_RADIUS; j < GRID_ROWS - MAX_RADIUS; j++) {
			// Initialize the maximal GICOV score to 0
			float max_GICOV = 0;

			// Iterate across each stencil
			for (k = 0; k < NCIRCLES; k++) {
				// Iterate across each sample point in the current stencil
				for (n = 0; n < NPOINTS; n++) {
					// Determine the x- and y-coordinates of the current sample point
					y = j + tY[k][n];
					x = i + tX[k][n];

					// Compute the combined gradient value at the current sample point
					Grad[n] = grad_x[y * GRID_COLS + x] * cos_angle[n] + grad_y[y * GRID_COLS + x] * sin_angle[n];
				}

				// Compute the mean gradient value across all sample points
				float sum = 0.0;
				for (n = 0; n < NPOINTS; n++) sum += Grad[n];
				float mean = sum / (float)NPOINTS;

				// Compute the variance of the gradient values
				float var = 0.0;
				for (n = 0; n < NPOINTS; n++) {
					sum = Grad[n] - mean;
					var += sum * sum;
				}
				var = var / (float)(NPOINTS - 1);

				// Keep track of the maximal GICOV value seen so far
				if (mean * mean / var > max_GICOV) {
					result[j * GRID_COLS + i] = mean / sqrt(var);
					max_GICOV = mean * mean / var;
				}
			}
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

static float gx[GRID_ROWS*GRID_COLS], gy[GRID_ROWS*GRID_COLS], res_ref[GRID_ROWS*GRID_COLS], res_dut[GRID_ROWS*GRID_COLS];

int main() {
    srand(42);
    for (int i = 0; i < GRID_ROWS*GRID_COLS; i++) {
        gx[i] = frand(-1.0f, 1.0f);
        gy[i] = frand(-1.0f, 1.0f);
        res_ref[i] = 12345.0f;
        res_dut[i] = 12345.0f;
    }
    workload_ref(res_ref, gx, gy);
    workload(res_dut, gx, gy);
    for (int r = MAX_RADIUS; r < GRID_ROWS - MAX_RADIUS; r++)
        for (int c = MAX_RADIUS; c < GRID_COLS - MAX_RADIUS; c++)
            chk_f("result", r * GRID_COLS + c, res_dut[r * GRID_COLS + c], res_ref[r * GRID_COLS + c]);
    if (errors) { printf("Failed with %d errors!\n", errors); return 1; }
    printf("Passed!\n");
    return 0;
}
