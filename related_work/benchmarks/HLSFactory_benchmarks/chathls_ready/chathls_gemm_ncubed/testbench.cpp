#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#define TYPE double

#define row_size 64
#define col_size 64
#define N row_size*col_size

#define MIN 0.
#define MAX 1.0

#define MAX_ITERATION 1

extern "C" {
void gemm_ncubed(TYPE m1[N], TYPE m2[N], TYPE prod[N]);
}

static TYPE v0_m1[N];
static TYPE v1_m2[N];
static TYPE v2_prod[N];

int main() {
  memset(v0_m1, 0, sizeof(v0_m1));
  memset(v1_m2, 0, sizeof(v1_m2));
  memset(v2_prod, 0, sizeof(v2_prod));
  gemm_ncubed(v0_m1, v1_m2, v2_prod);
  printf("PASS smoke\n");
  return 0;
}
