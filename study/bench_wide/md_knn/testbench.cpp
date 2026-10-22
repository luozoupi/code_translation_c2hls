/*
 * Strict Vitis HLS C-simulation testbench for md_knn.
 * Golden: md_ref() (same arithmetic as plain.cpp's md_kernel) on identical deterministic inputs (srand(42), no
 * self-neighbours so r^2 > 0). workload(force_x/y/z, position_x/y/z, NL): positions and NL are inputs; force_x/y/z are
 * pure outputs (assigned, never read first), poisoned with 12345.0 for ref and DUT. Every force element is checked,
 * double tol 1e-6*fmax(1,|ref|) (SPEC), NaN = error. Prints "Passed!" or "Failed with N errors!".
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "md.h"

static int errors = 0;
static void chk_d(const char *name, int i, double d, double r) {
    if (!isfinite(r) || !(fabs(d - r) <= 1e-6 * fmax(1.0, fabs(r)))) {   /* NaN/inf in DUT output (or a bad reference) is an error */
        if (errors < 10) printf("%s[%d]: got %.9g expected %.9g\n", name, i, d, r);
        errors++;
    }
}

/* Golden reference: plain-C MD kernel */
static void md_ref(TYPE force_x[nAtoms], TYPE force_y[nAtoms], TYPE force_z[nAtoms],
                   TYPE position_x[nAtoms], TYPE position_y[nAtoms], TYPE position_z[nAtoms],
                   int32_t NL[nAtoms * maxNeighbors]) {
    int i, j;
    for (i = 0; i < nAtoms; i++) {
        TYPE fx = 0, fy = 0, fz = 0;
        TYPE ix = position_x[i], iy = position_y[i], iz = position_z[i];
        for (j = 0; j < maxNeighbors; j++) {
            int jidx = NL[i * maxNeighbors + j];
            TYPE dx = ix - position_x[jidx];
            TYPE dy = iy - position_y[jidx];
            TYPE dz = iz - position_z[jidx];
            TYPE r2inv = 1.0 / (dx*dx + dy*dy + dz*dz);
            TYPE r6inv = r2inv * r2inv * r2inv;
            TYPE pot = r6inv * (lj1 * r6inv - lj2);
            TYPE f = r2inv * pot;
            fx += dx * f;
            fy += dy * f;
            fz += dz * f;
        }
        force_x[i] = fx;
        force_y[i] = fy;
        force_z[i] = fz;
    }
}

extern "C" void workload(TYPE* force_x, TYPE* force_y, TYPE* force_z,
              TYPE* position_x, TYPE* position_y, TYPE* position_z,
              int32_t* NL);

int main() {

    TYPE pos_x[nAtoms], pos_y[nAtoms], pos_z[nAtoms];
    int32_t NL[nAtoms * maxNeighbors];
    TYPE ref_fx[nAtoms], ref_fy[nAtoms], ref_fz[nAtoms];
    TYPE dut_fx[nAtoms], dut_fy[nAtoms], dut_fz[nAtoms];

    /* Generate deterministic random positions (avoid zero distance) */
    srand(42);
    for (int i = 0; i < nAtoms; i++) {
        pos_x[i] = (TYPE)(rand() % 1000) / 100.0 + 0.1;
        pos_y[i] = (TYPE)(rand() % 1000) / 100.0 + 0.1;
        pos_z[i] = (TYPE)(rand() % 1000) / 100.0 + 0.1;
    }
    /* Generate neighbor lists (ensure no self-neighbor) */
    for (int i = 0; i < nAtoms; i++) {
        for (int j = 0; j < maxNeighbors; j++) {
            int nb;
            do { nb = rand() % nAtoms; } while (nb == i);
            NL[i * maxNeighbors + j] = nb;
        }
    }

    /* Compute golden reference */
    for (int i = 0; i < nAtoms; i++) { ref_fx[i] = ref_fy[i] = ref_fz[i] = 12345.0; }   /* poison */
    md_ref(ref_fx, ref_fy, ref_fz, pos_x, pos_y, pos_z, NL);

    /* Call HLS design under test */
    for (int i = 0; i < nAtoms; i++) { dut_fx[i] = dut_fy[i] = dut_fz[i] = 12345.0; }   /* poison */
    workload(dut_fx, dut_fy, dut_fz, pos_x, pos_y, pos_z, NL);

    /* Compare outputs */
    for (int i = 0; i < nAtoms; i++) {
        chk_d("force_x", i, dut_fx[i], ref_fx[i]);
        chk_d("force_y", i, dut_fy[i], ref_fy[i]);
        chk_d("force_z", i, dut_fz[i], ref_fz[i]);
    }
    if (errors) { printf("Failed with %d errors!\n", errors); return 1; }
    printf("Passed!\n");
    return 0;
}
