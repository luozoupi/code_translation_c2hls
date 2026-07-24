#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include "ap_fixed.h"

typedef ap_fixed<32,16> t_ap_fixed;

extern "C" {
void gesummv(t_ap_fixed alpha, t_ap_fixed beta, t_ap_fixed A[ 30 + 0][30 + 0], t_ap_fixed B[ 30 + 0][30 + 0], t_ap_fixed tmp[ 30 + 0], t_ap_fixed x[ 30 + 0], t_ap_fixed y[ 30 + 0]);
}

static t_ap_fixed v2_A[ 30 + 0][30 + 0];
static t_ap_fixed v3_B[ 30 + 0][30 + 0];
static t_ap_fixed v4_tmp[ 30 + 0];
static t_ap_fixed v5_x[ 30 + 0];
static t_ap_fixed v6_y[ 30 + 0];

int main() {
  t_ap_fixed v0_alpha = (t_ap_fixed)32;
  t_ap_fixed v1_beta = (t_ap_fixed)32;
  memset(v2_A, 0, sizeof(v2_A));
  memset(v3_B, 0, sizeof(v3_B));
  memset(v4_tmp, 0, sizeof(v4_tmp));
  memset(v5_x, 0, sizeof(v5_x));
  memset(v6_y, 0, sizeof(v6_y));
  gesummv(v0_alpha, v1_beta, v2_A, v3_B, v4_tmp, v5_x, v6_y);
  printf("PASS smoke\n");
  return 0;
}
