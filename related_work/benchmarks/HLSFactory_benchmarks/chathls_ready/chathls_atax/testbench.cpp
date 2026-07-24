#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include "ap_fixed.h"

typedef ap_fixed<32,16> t_ap_fixed;

extern "C" {
void atax(t_ap_fixed A[ 38 + 0][42 + 0], t_ap_fixed x[ 42 + 0], t_ap_fixed y[ 42 + 0], t_ap_fixed tmp[ 38 + 0]);
}

static t_ap_fixed v0_A[ 38 + 0][42 + 0];
static t_ap_fixed v1_x[ 42 + 0];
static t_ap_fixed v2_y[ 42 + 0];
static t_ap_fixed v3_tmp[ 38 + 0];

int main() {
  memset(v0_A, 0, sizeof(v0_A));
  memset(v1_x, 0, sizeof(v1_x));
  memset(v2_y, 0, sizeof(v2_y));
  memset(v3_tmp, 0, sizeof(v3_tmp));
  atax(v0_A, v1_x, v2_y, v3_tmp);
  printf("PASS smoke\n");
  return 0;
}
