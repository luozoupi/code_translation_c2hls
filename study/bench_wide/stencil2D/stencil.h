#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define col_size 64
#define row_size 128
#define f_size 9
#define TYPE int32_t

void stencil(TYPE orig[row_size * col_size],
             TYPE sol[row_size * col_size],
             TYPE filter[f_size]);

struct bench_args_t {
    TYPE orig[row_size * col_size];
    TYPE sol[row_size * col_size];
    TYPE filter[f_size];
};
