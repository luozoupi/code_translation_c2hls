/*
 * Strict Vitis HLS C-simulation testbench for fft (512-point, double).
 * Golden: plain.cpp (twiddles8, loady8, fft1D_512, workload) copied below with _ref suffix and run on identical
 * copies of the deterministic input (sum of two sinusoids in x, a sinusoid in y). workload(work_x, work_y): both arrays
 * are in-out (real input, FFT result written back), so there is no pure output to poison; every element of work_x and
 * work_y is checked, double tol 1e-6*fmax(1,|ref|) (SPEC), NaN = error. Parseval is also checked as a sanity invariant.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "fft.h"

#define FFT_SIZE 512

extern "C" void workload(TYPE* work_x, TYPE* work_y);

/* ---- reference: copy of plain.cpp ---- */

/* Same as plain.cpp but with extern "C" workload_ref wrapper and HLS INTERFACE pragmas.
 * The full FFT kernel is identical to plain.cpp. */

#define THREADS 64
#define cmplx_M_x(a_x, a_y, b_x, b_y) (a_x*b_x - a_y*b_y)
#define cmplx_M_y(a_x, a_y, b_x, b_y) (a_x*b_y + a_y*b_x)
#define cmplx_mul_x(a_x, a_y, b_x, b_y) (a_x*b_x - a_y*b_y)
#define cmplx_mul_y(a_x, a_y, b_x, b_y) (a_x*b_y + a_y*b_x)
#define cmplx_add_x(a_x, b_x) (a_x + b_x)
#define cmplx_add_y(a_y, b_y) (a_y + b_y)
#define cmplx_sub_x(a_x, b_x) (a_x - b_x)
#define cmplx_sub_y(a_y, b_y) (a_y - b_y)
#define cm_fl_mul_x(a_x, b) (b*a_x)
#define cm_fl_mul_y(a_y, b) (b*a_y)

static void twiddles8_ref(TYPE a_x[8], TYPE a_y[8], int i, int n) {
    int reversed8[8] = {0,4,2,6,1,5,3,7};
    int j;
    TYPE phi, tmp, phi_x, phi_y;
    twiddles: for (j = 1; j < 8; j++) {
        phi = ((-2*PI*reversed8[j]/n)*i);
        phi_x = cos(phi);
        phi_y = sin(phi);
        tmp = a_x[j];
        a_x[j] = cmplx_M_x(a_x[j], a_y[j], phi_x, phi_y);
        a_y[j] = cmplx_M_y(tmp, a_y[j], phi_x, phi_y);
    }
}

#define FF2(a0_x, a0_y, a1_x, a1_y){ \
    TYPE c0_x = *a0_x; TYPE c0_y = *a0_y; \
    *a0_x = cmplx_add_x(c0_x, *a1_x); *a0_y = cmplx_add_y(c0_y, *a1_y); \
    *a1_x = cmplx_sub_x(c0_x, *a1_x); *a1_y = cmplx_sub_y(c0_y, *a1_y); \
}

#define FFT4(a0_x, a0_y, a1_x, a1_y, a2_x, a2_y, a3_x, a3_y){ \
    TYPE exp_1_44_x, exp_1_44_y, tmp; \
    exp_1_44_x = 0.0; exp_1_44_y = -1.0; \
    FF2(a0_x, a0_y, a2_x, a2_y); FF2(a1_x, a1_y, a3_x, a3_y); \
    tmp = *a3_x; *a3_x = *a3_x*exp_1_44_x-*a3_y*exp_1_44_y; \
    *a3_y = tmp*exp_1_44_y - *a3_y*exp_1_44_x; \
    FF2(a0_x, a0_y, a1_x, a1_y); FF2(a2_x, a2_y, a3_x, a3_y); \
}

#define FFT8(a_x, a_y) { \
    TYPE exp_1_8_x=1, exp_1_4_x=0, exp_3_8_x=-1; \
    TYPE exp_1_8_y=-1, exp_1_4_y=-1, exp_3_8_y=-1; \
    TYPE tmp_1; \
    FF2(&a_x[0],&a_y[0],&a_x[4],&a_y[4]); FF2(&a_x[1],&a_y[1],&a_x[5],&a_y[5]); \
    FF2(&a_x[2],&a_y[2],&a_x[6],&a_y[6]); FF2(&a_x[3],&a_y[3],&a_x[7],&a_y[7]); \
    tmp_1=a_x[5]; a_x[5]=cm_fl_mul_x(cmplx_mul_x(a_x[5],a_y[5],exp_1_8_x,exp_1_8_y),M_SQRT1_2); \
    a_y[5]=cm_fl_mul_y(cmplx_mul_y(tmp_1,a_y[5],exp_1_8_x,exp_1_8_y),M_SQRT1_2); \
    tmp_1=a_x[6]; a_x[6]=cmplx_mul_x(a_x[6],a_y[6],exp_1_4_x,exp_1_4_y); \
    a_y[6]=cmplx_mul_y(tmp_1,a_y[6],exp_1_4_x,exp_1_4_y); \
    tmp_1=a_x[7]; a_x[7]=cm_fl_mul_x(cmplx_mul_x(a_x[7],a_y[7],exp_3_8_x,exp_3_8_y),M_SQRT1_2); \
    a_y[7]=cm_fl_mul_y(cmplx_mul_y(tmp_1,a_y[7],exp_3_8_x,exp_3_8_y),M_SQRT1_2); \
    FFT4(&a_x[0],&a_y[0],&a_x[1],&a_y[1],&a_x[2],&a_y[2],&a_x[3],&a_y[3]); \
    FFT4(&a_x[4],&a_y[4],&a_x[5],&a_y[5],&a_x[6],&a_y[6],&a_x[7],&a_y[7]); \
}

static void loady8_ref(TYPE a_y[], TYPE x[], int offset, int sx) {
    for (int k = 0; k < 8; k++) a_y[k] = x[k*sx+offset];
}

static void fft1D_512_ref(TYPE work_x[512], TYPE work_y[512]) {
    int tid, hi, lo, stride;
    int reversed[] = {0,4,2,6,1,5,3,7};
    TYPE DATA_x[THREADS*8], DATA_y[THREADS*8];
    TYPE data_x[8], data_y[8];
    TYPE smem[8*8*9];
    stride = THREADS;

    loop1: for (tid = 0; tid < THREADS; tid++) {
        for (int k = 0; k < 8; k++) { data_x[k] = work_x[k*stride+tid]; data_y[k] = work_y[k*stride+tid]; }
        FFT8(data_x, data_y);
        twiddles8_ref(data_x, data_y, tid, 512);
        for (int k = 0; k < 8; k++) { DATA_x[tid*8+k] = data_x[k]; DATA_y[tid*8+k] = data_y[k]; }
    }
    int sx, offset;
    sx = 66;
    loop2: for (tid = 0; tid < 64; tid++) { hi=tid>>3; lo=tid&7; offset=hi*8+lo;
        smem[0*sx+offset]=DATA_x[tid*8+0]; smem[4*sx+offset]=DATA_x[tid*8+1];
        smem[1*sx+offset]=DATA_x[tid*8+4]; smem[5*sx+offset]=DATA_x[tid*8+5];
        smem[2*sx+offset]=DATA_x[tid*8+2]; smem[6*sx+offset]=DATA_x[tid*8+3];
        smem[3*sx+offset]=DATA_x[tid*8+6]; smem[7*sx+offset]=DATA_x[tid*8+7]; }
    sx = 8;
    loop3: for (tid = 0; tid < 64; tid++) { hi=tid>>3; lo=tid&7; offset=lo*66+hi;
        DATA_x[tid*8+0]=smem[0*sx+offset]; DATA_x[tid*8+4]=smem[4*sx+offset];
        DATA_x[tid*8+1]=smem[1*sx+offset]; DATA_x[tid*8+5]=smem[5*sx+offset];
        DATA_x[tid*8+2]=smem[2*sx+offset]; DATA_x[tid*8+6]=smem[6*sx+offset];
        DATA_x[tid*8+3]=smem[3*sx+offset]; DATA_x[tid*8+7]=smem[7*sx+offset]; }
    sx = 66;
    loop4: for (tid = 0; tid < 64; tid++) { hi=tid>>3; lo=tid&7; offset=hi*8+lo;
        smem[0*sx+offset]=DATA_y[tid*8+0]; smem[4*sx+offset]=DATA_y[tid*8+1];
        smem[1*sx+offset]=DATA_y[tid*8+4]; smem[5*sx+offset]=DATA_y[tid*8+5];
        smem[2*sx+offset]=DATA_y[tid*8+2]; smem[6*sx+offset]=DATA_y[tid*8+3];
        smem[3*sx+offset]=DATA_y[tid*8+6]; smem[7*sx+offset]=DATA_y[tid*8+7]; }
    loop5: for (tid = 0; tid < 64; tid++) { hi=tid>>3; lo=tid&7;
        loady8_ref(data_y, smem, lo*66+hi, 8);
        for (int k = 0; k < 8; k++) DATA_y[tid*8+k] = data_y[k]; }
    loop6: for (tid = 0; tid < 64; tid++) {
        for (int k = 0; k < 8; k++) { data_x[k]=DATA_x[tid*8+k]; data_y[k]=DATA_y[tid*8+k]; }
        FFT8(data_x, data_y); hi=tid>>3;
        twiddles8_ref(data_x, data_y, hi, 64);
        for (int k = 0; k < 8; k++) { DATA_x[tid*8+k]=data_x[k]; DATA_y[tid*8+k]=data_y[k]; } }
    sx = 72;
    loop7: for (tid = 0; tid < 64; tid++) { hi=tid>>3; lo=tid&7; offset=hi*8+lo;
        smem[0*sx+offset]=DATA_x[tid*8+0]; smem[4*sx+offset]=DATA_x[tid*8+1];
        smem[1*sx+offset]=DATA_x[tid*8+4]; smem[5*sx+offset]=DATA_x[tid*8+5];
        smem[2*sx+offset]=DATA_x[tid*8+2]; smem[6*sx+offset]=DATA_x[tid*8+3];
        smem[3*sx+offset]=DATA_x[tid*8+6]; smem[7*sx+offset]=DATA_x[tid*8+7]; }
    sx = 8;
    loop8: for (tid = 0; tid < 64; tid++) { hi=tid>>3; lo=tid&7; offset=hi*72+lo;
        DATA_x[tid*8+0]=smem[0*sx+offset]; DATA_x[tid*8+4]=smem[4*sx+offset];
        DATA_x[tid*8+1]=smem[1*sx+offset]; DATA_x[tid*8+5]=smem[5*sx+offset];
        DATA_x[tid*8+2]=smem[2*sx+offset]; DATA_x[tid*8+6]=smem[6*sx+offset];
        DATA_x[tid*8+3]=smem[3*sx+offset]; DATA_x[tid*8+7]=smem[7*sx+offset]; }
    sx = 72;
    loop9: for (tid = 0; tid < 64; tid++) { hi=tid>>3; lo=tid&7; offset=hi*8+lo;
        smem[0*sx+offset]=DATA_y[tid*8+0]; smem[4*sx+offset]=DATA_y[tid*8+1];
        smem[1*sx+offset]=DATA_y[tid*8+4]; smem[5*sx+offset]=DATA_y[tid*8+5];
        smem[2*sx+offset]=DATA_y[tid*8+2]; smem[6*sx+offset]=DATA_y[tid*8+3];
        smem[3*sx+offset]=DATA_y[tid*8+6]; smem[7*sx+offset]=DATA_y[tid*8+7]; }
    loop10: for (tid = 0; tid < 64; tid++) { hi=tid>>3; lo=tid&7;
        loady8_ref(data_y, smem, hi*72+lo, 8);
        for (int k = 0; k < 8; k++) DATA_y[tid*8+k] = data_y[k]; }
    loop11: for (tid = 0; tid < 64; tid++) {
        for (int k = 0; k < 8; k++) { data_x[k]=DATA_x[tid*8+k]; data_y[k]=DATA_y[tid*8+k]; }
        FFT8(data_x, data_y);
        for (int k = 0; k < 8; k++) { work_x[k*stride+tid]=data_x[reversed[k]]; work_y[k*stride+tid]=data_y[reversed[k]]; }
    }
}


static void workload_ref(TYPE* work_x, TYPE* work_y) {

    TYPE l_x[512], l_y[512];
    int i;
    for (i = 0; i < 512; i++) { l_x[i] = work_x[i]; l_y[i] = work_y[i]; }
    fft1D_512_ref(l_x, l_y);
    for (i = 0; i < 512; i++) { work_x[i] = l_x[i]; work_y[i] = l_y[i]; }
}
/* ---- end reference ---- */

static int errors = 0;
static void chk_d(const char *name, int i, double d, double r) {
    if (!isfinite(r) || !(fabs(d - r) <= 1e-6 * fmax(1.0, fabs(r)))) {   /* NaN/inf in DUT output (or a bad reference) is an error */
        if (errors < 10) printf("%s[%d]: got %.9g expected %.9g\n", name, i, d, r);
        errors++;
    }
}

int main() {
    TYPE ref_x[FFT_SIZE], ref_y[FFT_SIZE];
    TYPE dut_x[FFT_SIZE], dut_y[FFT_SIZE];
    TYPE in_energy = 0;

    /* deterministic test signal */
    for (int i = 0; i < FFT_SIZE; i++) {
        ref_x[i] = cos(2.0 * PI * 10.0 * i / FFT_SIZE) + 0.5 * cos(2.0 * PI * 50.0 * i / FFT_SIZE);
        ref_y[i] = 0.3 * sin(2.0 * PI * 7.0 * i / FFT_SIZE);
        dut_x[i] = ref_x[i];
        dut_y[i] = ref_y[i];
        in_energy += ref_x[i]*ref_x[i] + ref_y[i]*ref_y[i];
    }

    workload_ref(ref_x, ref_y);
    workload(dut_x, dut_y);

    for (int i = 0; i < FFT_SIZE; i++) chk_d("work_x", i, dut_x[i], ref_x[i]);
    for (int i = 0; i < FFT_SIZE; i++) chk_d("work_y", i, dut_y[i], ref_y[i]);

    /* Parseval: sum|X|^2 = N * sum|x|^2 (checks the reference itself too) */
    TYPE energy = 0;
    for (int i = 0; i < FFT_SIZE; i++) energy += dut_x[i]*dut_x[i] + dut_y[i]*dut_y[i];
    if (fabs(energy - FFT_SIZE * in_energy) > 1e-6 * fmax(1.0, FFT_SIZE * in_energy)) {
        printf("Parseval mismatch: %.9g vs %.9g\n", energy, FFT_SIZE * in_energy);
        errors++;
    }
    if (errors) { printf("Failed with %d errors!\n", errors); return 1; }
    printf("Passed!\n");
    return 0;
}
