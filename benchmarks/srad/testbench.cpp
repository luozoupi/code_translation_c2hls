/*
 * Vitis HLS C-simulation testbench for SRAD (Speckle Reducing Anisotropic Diffusion).
 * Replicates the tiled kernel2 algorithm as golden reference,
 * then compares against HLS workload() output.
 * Returns 0 on success, 1 on mismatch.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "srad.h"

extern "C" void workload(float J[(ROWS+3)*COLS], float Jout[(ROWS+3)*COLS]);

/* srad helper functions (matching kernel) */
static float srad_core1_ref(float dN, float dS, float dW, float dE,
                            float Jc, float q0sqr)
{
    float G2 = (dN*dN + dS*dS + dW*dW + dE*dE) / (Jc*Jc);
    float L  = (dN + dS + dW + dE) / Jc;
    float num  = (0.5f*G2) - ((1.0f/16.0f)*(L*L));
    float den  = 1.0f + (0.25f*L);
    float qsqr = num / (den*den);
    den = (qsqr - q0sqr) / (q0sqr * (1.0f + q0sqr));
    float c = 1.0f / (1.0f + den);
    return c;
}

static float srad_core2_ref(float dN, float dS, float dW, float dE,
                            float cN, float cS, float cW, float cE,
                            float J_val)
{
    float D = cN*dN + cS*dS + cW*dW + cE*dE;
    return J_val + 0.25f * LAMBDA * D;
}

/*
 * Golden reference: simplified naive SRAD iteration.
 * Computes directional derivatives, diffusion coefficients, then update.
 * Processes one tile at a time to match the kernel's tiled approach.
 */
static void srad_kernel2_ref(float* J, float* Jout, float q0sqr, int tile)
{
    /* J has (TILE_ROWS+3)*COLS elements, Jout has TILE_ROWS*COLS */
    /* J[row][col] where row 0..TILE_ROWS+2, the "center" starts at row 1 */

    int TR = TILE_ROWS;
    float dN[(TILE_ROWS+1)*COLS];
    float dS[(TILE_ROWS+1)*COLS];
    float dW[(TILE_ROWS+1)*COLS];
    float dE[(TILE_ROWS+1)*COLS];
    float c[(TILE_ROWS+1)*COLS];

    /* Compute directional derivatives and diffusion coefficients */
    for (int i = 0; i < TR + 1; i++) {
        for (int j = 0; j < COLS; j++) {
            int idx = i * COLS + j;
            /* Center pixel is at J[(i+1)*COLS + j] */
            float Jc = J[(i+1)*COLS + j];

            /* North neighbor */
            float Jn;
            if (tile == 0 && i == 0)
                Jn = Jc;  /* top border: replicate */
            else
                Jn = J[i*COLS + j];

            /* South neighbor */
            float Js;
            if (tile == (ROWS/TR - 1) && i == TR)
                Js = Jc;
            else
                Js = J[(i+2)*COLS + j];

            /* West neighbor */
            float Jw = (j == 0) ? Jc : J[(i+1)*COLS + j - 1];
            /* East neighbor */
            float Je = (j == COLS-1) ? Jc : J[(i+1)*COLS + j + 1];

            dN[idx] = Jn - Jc;
            dS[idx] = Js - Jc;
            dW[idx] = Jw - Jc;
            dE[idx] = Je - Jc;

            float c_tmp = srad_core1_ref(dN[idx], dS[idx], dW[idx], dE[idx], Jc, q0sqr);
            if (c_tmp < 0.0f) c[idx] = 0.0f;
            else if (c_tmp > 1.0f) c[idx] = 1.0f;
            else c[idx] = c_tmp;
        }
    }

    /* Compute output using diffusion update */
    for (int i = 0; i < TR; i++) {
        for (int j = 0; j < COLS; j++) {
            int idx = i * COLS + j;
            float cN_val = c[idx];
            float cS_val;
            if (tile == (ROWS/TR - 1) && i == TR - 1)
                cS_val = c[idx];
            else
                cS_val = c[(i+1)*COLS + j];

            float cW_val = c[idx];
            float cE_val = (j == COLS-1) ? c[idx] : c[i*COLS + j + 1];

            Jout[idx] = srad_core2_ref(dN[idx], dS[idx], dW[idx], dE[idx],
                                       cN_val, cS_val, cW_val, cE_val,
                                       J[COLS + idx]);
        }
    }
}

static void workload_ref(float* J, float* Jout)
{
    float J_buf[(TILE_ROWS+3)*COLS];
    float Jout_buf[TILE_ROWS*COLS];
    float v0sqr = 0.0870038941502571f;

    for (int iter = 0; iter < NITER/2; iter++) {
        for (int t = 0; t < ROWS/TILE_ROWS; t++) {
            memcpy(J_buf, J + t*TILE_ROWS*COLS, (TILE_ROWS+3)*COLS*sizeof(float));
            srad_kernel2_ref(J_buf, Jout_buf, v0sqr, t);
            memcpy(Jout + (t*TILE_ROWS+1)*COLS, Jout_buf, TILE_ROWS*COLS*sizeof(float));
        }
        for (int t = 0; t < ROWS/TILE_ROWS; t++) {
            memcpy(J_buf, Jout + t*TILE_ROWS*COLS, (TILE_ROWS+3)*COLS*sizeof(float));
            srad_kernel2_ref(J_buf, Jout_buf, v0sqr, t);
            memcpy(J + (t*TILE_ROWS+1)*COLS, Jout_buf, TILE_ROWS*COLS*sizeof(float));
        }
    }
}

int main() {
    int errors = 0;
    int total = (ROWS+3) * COLS;
    srand(42);

    float* J_ref   = new float[total];
    float* J_dut   = new float[total];
    float* Out_ref = new float[total];
    float* Out_dut = new float[total];

    /* Generate deterministic image data (positive values for valid diffusion) */
    for (int i = 0; i < total; i++)
        J_ref[i] = 50.0f + (float)(rand() % 200) / 10.0f;

    memcpy(J_dut, J_ref, total * sizeof(float));
    memcpy(Out_ref, J_ref, total * sizeof(float));
    memcpy(Out_dut, J_ref, total * sizeof(float));

    /* Compute golden reference */
    workload_ref(J_ref, Out_ref);

    /* Call HLS DUT */
    workload(J_dut, Out_dut);

    /* Compare interior rows (rows 1..ROWS) where output is written */
    float tol = 1e-2f;
    int mismatches = 0;
    for (int r = 1; r <= ROWS; r++) {
        for (int c_col = 0; c_col < COLS; c_col++) {
            int idx = r * COLS + c_col;
            float diff = fabsf(Out_ref[idx] - Out_dut[idx]);
            float scale = fabsf(Out_ref[idx]) + 1.0f;
            if (diff > tol * scale) {
                if (mismatches < 5) {
                    printf("  mismatch[%d,%d]: ref=%.6f dut=%.6f\n",
                           r, c_col, Out_ref[idx], Out_dut[idx]);
                }
                mismatches++;
            }
        }
    }
    if (mismatches > 0) {
        printf("FAIL: Jout — %d mismatches out of %d\n", mismatches, ROWS * COLS);
        errors++;
    }

    if (errors == 0) {
        printf("PASS: srad testbench — all outputs match golden reference\n");
    } else {
        printf("FAIL: srad testbench — %d error(s)\n", errors);
    }

    delete[] J_ref; delete[] J_dut;
    delete[] Out_ref; delete[] Out_dut;
    return (errors > 0) ? 1 : 0;
}
