#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include "ap_fixed.h"

typedef ap_fixed<32,16> t_ap_fixed;

extern "C" {
void bicg(t_ap_fixed A[ 42 + 0][38 + 0], t_ap_fixed s[ 38 + 0], t_ap_fixed q[ 42 + 0], t_ap_fixed p[ 38 + 0], t_ap_fixed r[ 42 + 0]);
}

static t_ap_fixed v0_A[ 42 + 0][38 + 0];
static t_ap_fixed v1_s[ 38 + 0];
static t_ap_fixed v2_q[ 42 + 0];
static t_ap_fixed v3_p[ 38 + 0];
static t_ap_fixed v4_r[ 42 + 0];

int main() {
  memset(v0_A, 0, sizeof(v0_A));
  memset(v1_s, 0, sizeof(v1_s));
  memset(v2_q, 0, sizeof(v2_q));
  memset(v3_p, 0, sizeof(v3_p));
  memset(v4_r, 0, sizeof(v4_r));
  bicg(v0_A, v1_s, v2_q, v3_p, v4_r);
  printf("PASS smoke\n");
  return 0;
}
