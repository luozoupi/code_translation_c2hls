#ifndef SPMV_H_
#define SPMV_H_
#include "params.h"

void spmv(
    int Ap[num_rows],
    int Aj[num_rows],
    int Ax[num_rows],
    int x[num_rows],
    int y[num_rows]);

#endif
