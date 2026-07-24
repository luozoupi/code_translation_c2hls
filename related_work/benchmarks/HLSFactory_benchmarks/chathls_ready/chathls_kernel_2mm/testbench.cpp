#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

extern "C" {
void kernel_2mm(double alpha, double beta, double tmp[40][50], double A[40][70], double B[70][50], double C[50][80], double D[40][80]);
}

static double v2_tmp[40][50];
static double v3_A[40][70];
static double v4_B[70][50];
static double v5_C[50][80];
static double v6_D[40][80];

int main() {
  double v0_alpha = (double)32;
  double v1_beta = (double)32;
  memset(v2_tmp, 0, sizeof(v2_tmp));
  memset(v3_A, 0, sizeof(v3_A));
  memset(v4_B, 0, sizeof(v4_B));
  memset(v5_C, 0, sizeof(v5_C));
  memset(v6_D, 0, sizeof(v6_D));
  kernel_2mm(v0_alpha, v1_beta, v2_tmp, v3_A, v4_B, v5_C, v6_D);
  printf("PASS smoke\n");
  return 0;
}
