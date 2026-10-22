#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define NNZ 1666
#define N 494
#define TYPE double

void spmv(TYPE val[NNZ], int32_t cols[NNZ], int32_t rowDelimiters[N + 1],
          TYPE vec[N], TYPE out[N]);

struct bench_args_t {
    TYPE val[NNZ];
    int32_t cols[NNZ];
    int32_t rowDelimiters[N + 1];
    TYPE vec[N];
    TYPE out[N];
};
