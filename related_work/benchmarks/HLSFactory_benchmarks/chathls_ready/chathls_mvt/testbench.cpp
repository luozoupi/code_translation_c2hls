#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include "ap_fixed.h"

typedef ap_fixed<32,16> t_ap_fixed;

extern "C" {
void mvt(t_ap_fixed x1[ 40 + 0], t_ap_fixed x2[ 40 + 0], t_ap_fixed y_1[ 40 + 0], t_ap_fixed y_2[ 40 + 0], t_ap_fixed A[ 40 + 0][40 + 0]);
}

static t_ap_fixed v0_x1[ 40 + 0];
static t_ap_fixed v1_x2[ 40 + 0];
static t_ap_fixed v2_y_1[ 40 + 0];
static t_ap_fixed v3_y_2[ 40 + 0];
static t_ap_fixed v4_A[ 40 + 0][40 + 0];

int main() {
  memset(v0_x1, 0, sizeof(v0_x1));
  memset(v1_x2, 0, sizeof(v1_x2));
  memset(v2_y_1, 0, sizeof(v2_y_1));
  memset(v3_y_2, 0, sizeof(v3_y_2));
  memset(v4_A, 0, sizeof(v4_A));
  mvt(v0_x1, v1_x2, v2_y_1, v3_y_2, v4_A);
  printf("PASS smoke\n");
  return 0;
}
