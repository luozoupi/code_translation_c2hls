#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include "ap_fixed.h"

typedef ap_fixed<32,16> t_ap_fixed;

extern "C" {
void gemm(t_ap_fixed alpha, t_ap_fixed beta, t_ap_fixed C[ 20 + 0][25 + 0], t_ap_fixed A[ 20 + 0][30 + 0], t_ap_fixed B[ 30 + 0][25 + 0]);
}

static t_ap_fixed v2_C[ 20 + 0][25 + 0];
static t_ap_fixed v3_A[ 20 + 0][30 + 0];
static t_ap_fixed v4_B[ 30 + 0][25 + 0];

int main() {
  t_ap_fixed v0_alpha = (t_ap_fixed)32;
  t_ap_fixed v1_beta = (t_ap_fixed)32;
  memset(v2_C, 0, sizeof(v2_C));
  memset(v3_A, 0, sizeof(v3_A));
  memset(v4_B, 0, sizeof(v4_B));
  gemm(v0_alpha, v1_beta, v2_C, v3_A, v4_B);
  printf("PASS smoke\n");
  return 0;
}
