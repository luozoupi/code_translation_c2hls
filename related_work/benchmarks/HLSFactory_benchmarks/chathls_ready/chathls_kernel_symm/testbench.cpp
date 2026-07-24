#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

extern "C" {
void kernel_symm(double alpha, double beta, double C[60][80], double A[60][60], double B[60][80]);
}

static double v2_C[60][80];
static double v3_A[60][60];
static double v4_B[60][80];

int main() {
  double v0_alpha = (double)32;
  double v1_beta = (double)32;
  memset(v2_C, 0, sizeof(v2_C));
  memset(v3_A, 0, sizeof(v3_A));
  memset(v4_B, 0, sizeof(v4_B));
  kernel_symm(v0_alpha, v1_beta, v2_C, v3_A, v4_B);
  printf("PASS smoke\n");
  return 0;
}
