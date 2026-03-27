/*
 * Vitis HLS C-simulation testbench for fft (512-point) benchmark.
 * Generates a known signal, runs golden reference and HLS workload(),
 * compares FFT output. Also checks Parseval's theorem (energy conservation).
 * Returns 0 on success, 1 on mismatch.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "fft.h"

#define EPSILON 1.0e-4
#define FFT_SIZE 512

extern "C" void workload(TYPE* work_x, TYPE* work_y);

int main() {
    int errors = 0;

    TYPE ref_x[FFT_SIZE], ref_y[FFT_SIZE];
    TYPE dut_x[FFT_SIZE], dut_y[FFT_SIZE];

    /* Generate deterministic test signal: sum of two sinusoids */
    for (int i = 0; i < FFT_SIZE; i++) {
        ref_x[i] = cos(2.0 * PI * 10.0 * i / FFT_SIZE) +
                    0.5 * cos(2.0 * PI * 50.0 * i / FFT_SIZE);
        ref_y[i] = 0.0;
        dut_x[i] = ref_x[i];
        dut_y[i] = 0.0;
    }

    /* Compute golden reference */
    fft1D_512(ref_x, ref_y);

    /* Call HLS design under test */
    workload(dut_x, dut_y);

    /* Compare FFT outputs */
    for (int i = 0; i < FFT_SIZE; i++) {
        TYPE dx = fabs(dut_x[i] - ref_x[i]);
        TYPE dy = fabs(dut_y[i] - ref_y[i]);
        if (dx > EPSILON || dy > EPSILON) {
            if (errors < 10)
                printf("FAIL: bin[%d] dut=(%.6f,%.6f) ref=(%.6f,%.6f)\n",
                       i, dut_x[i], dut_y[i], ref_x[i], ref_y[i]);
            errors++;
        }
    }

    /* Parseval's check: energy should be preserved */
    TYPE energy_dut = 0, energy_ref = 0;
    for (int i = 0; i < FFT_SIZE; i++) {
        energy_dut += dut_x[i]*dut_x[i] + dut_y[i]*dut_y[i];
        energy_ref += ref_x[i]*ref_x[i] + ref_y[i]*ref_y[i];
    }
    if (fabs(energy_dut - energy_ref) > 1.0) {
        printf("FAIL: energy mismatch (dut=%.2f, ref=%.2f)\n", energy_dut, energy_ref);
        errors++;
    }

    if (errors == 0) {
        printf("PASS: fft testbench — all %d bins match, energy conserved\n", FFT_SIZE);
    } else {
        printf("FAIL: fft testbench — %d error(s)\n", errors);
    }

    return (errors > 0) ? 1 : 0;
}
