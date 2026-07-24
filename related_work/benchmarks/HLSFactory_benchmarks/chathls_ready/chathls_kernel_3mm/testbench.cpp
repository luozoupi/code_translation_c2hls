#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

extern "C" {
void kernel_3mm(double E[40][50], double A[40][60], double B[60][50], double F[50][70], double C[50][80], double D[80][70], double G[40][70]);
}

static double v0_E[40][50];
static double v1_A[40][60];
static double v2_B[60][50];
static double v3_F[50][70];
static double v4_C[50][80];
static double v5_D[80][70];
static double v6_G[40][70];

int main() {
  memset(v0_E, 0, sizeof(v0_E));
  memset(v1_A, 0, sizeof(v1_A));
  memset(v2_B, 0, sizeof(v2_B));
  memset(v3_F, 0, sizeof(v3_F));
  memset(v4_C, 0, sizeof(v4_C));
  memset(v5_D, 0, sizeof(v5_D));
  memset(v6_G, 0, sizeof(v6_G));
  kernel_3mm(v0_E, v1_A, v2_B, v3_F, v4_C, v5_D, v6_G);
  printf("PASS smoke\n");
  return 0;
}
