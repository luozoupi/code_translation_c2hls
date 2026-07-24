#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include "ap_fixed.h"

typedef ap_fixed<32,16> t_ap_fixed;

extern "C" {
void covariance(t_ap_fixed float_n, t_ap_fixed data[ 32 + 0][28 + 0], t_ap_fixed cov[ 28 + 0][28 + 0], t_ap_fixed mean[ 28 + 0]);
}

static t_ap_fixed v1_data[ 32 + 0][28 + 0];
static t_ap_fixed v2_cov[ 28 + 0][28 + 0];
static t_ap_fixed v3_mean[ 28 + 0];

int main() {
  t_ap_fixed v0_float_n = (t_ap_fixed)32;
  memset(v1_data, 0, sizeof(v1_data));
  memset(v2_cov, 0, sizeof(v2_cov));
  memset(v3_mean, 0, sizeof(v3_mean));
  covariance(v0_float_n, v1_data, v2_cov, v3_mean);
  printf("PASS smoke\n");
  return 0;
}
