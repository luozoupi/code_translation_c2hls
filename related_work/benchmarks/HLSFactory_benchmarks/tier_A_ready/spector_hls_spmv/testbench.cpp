#include "spmv.h"
#include "params.h"
#include <stdio.h>

int main() {
    static int Ap[num_rows + 1];
    static int Aj[num_rows];
    static int Ax[num_rows];
    static int x[num_rows];
    static int y[num_rows];
    for (int i = 0; i < num_rows + 1; i++)
        Ap[i] = i;
    for (int i = 0; i < num_rows; i++) {
        Aj[i] = i;
        Ax[i] = 1;
        x[i] = 1;
        y[i] = 0;
    }
    spmv(Ap, Aj, Ax, x, y);
    printf("spmv csim done\n");
    return 0;
}
