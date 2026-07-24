#include "mergesort.h"
#include "params.h"
#include <stdio.h>

int main() {
    static int in[no_size];
    static int out[no_size];
    for (int i = 0; i < no_size; i++)
        in[i] = no_size - 1 - i;
    for (int i = 0; i < no_size; i++)
        out[i] = 0;
    mergesort(in, out);
    printf("mergesort csim done\n");
    return 0;
}
