#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

extern "C" {
void kernel_syrk(double alpha, double beta, double C[80][80], double A[80][60]);
}

static double v2_C[80][80];
static double v3_A[80][60];

int main() {
  double v0_alpha = (double)32;
  double v1_beta = (double)32;
  memset(v2_C, 0, sizeof(v2_C));
  memset(v3_A, 0, sizeof(v3_A));
  kernel_syrk(v0_alpha, v1_beta, v2_C, v3_A);
  printf("PASS smoke\n");
  return 0;
}
