#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>

#define SIZE 2048
#define TYPE int32_t
#define TYPE_MAX INT32_MAX

void ms_mergesort(TYPE a[SIZE]);

struct bench_args_t {
    TYPE a[SIZE];
};
