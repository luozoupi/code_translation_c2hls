#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#define TYPE double
#define PI 3.1415926535
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440f
#endif

void fft1D_512(TYPE work_x[512], TYPE work_y[512]);

struct bench_args_t {
    TYPE work_x[512];
    TYPE work_y[512];
};
