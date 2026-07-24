#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include "ap_fixed.h"

typedef ap_fixed<16, 5> data_t;

extern "C" {
void transformer(data_t DRAM_attn_input[8][32], data_t DRAM_weights_q[32][32], data_t DRAM_weights_k[32][32], data_t DRAM_weights_v[32][32], data_t DRAM_layer_norm_weights_1[2][32], data_t DRAM_FF_weights_1[128][32], data_t DRAM_FF_weights_2[32][128], data_t DRAM_layer_norm_weights_2[2][32], data_t DRAM_output[8][32]);
}

static data_t v0_DRAM_attn_input[8][32];
static data_t v1_DRAM_weights_q[32][32];
static data_t v2_DRAM_weights_k[32][32];
static data_t v3_DRAM_weights_v[32][32];
static data_t v4_DRAM_layer_norm_weights_1[2][32];
static data_t v5_DRAM_FF_weights_1[128][32];
static data_t v6_DRAM_FF_weights_2[32][128];
static data_t v7_DRAM_layer_norm_weights_2[2][32];
static data_t v8_DRAM_output[8][32];

int main() {
  memset(v0_DRAM_attn_input, 0, sizeof(v0_DRAM_attn_input));
  memset(v1_DRAM_weights_q, 0, sizeof(v1_DRAM_weights_q));
  memset(v2_DRAM_weights_k, 0, sizeof(v2_DRAM_weights_k));
  memset(v3_DRAM_weights_v, 0, sizeof(v3_DRAM_weights_v));
  memset(v4_DRAM_layer_norm_weights_1, 0, sizeof(v4_DRAM_layer_norm_weights_1));
  memset(v5_DRAM_FF_weights_1, 0, sizeof(v5_DRAM_FF_weights_1));
  memset(v6_DRAM_FF_weights_2, 0, sizeof(v6_DRAM_FF_weights_2));
  memset(v7_DRAM_layer_norm_weights_2, 0, sizeof(v7_DRAM_layer_norm_weights_2));
  memset(v8_DRAM_output, 0, sizeof(v8_DRAM_output));
  transformer(v0_DRAM_attn_input, v1_DRAM_weights_q, v2_DRAM_weights_k, v3_DRAM_weights_v, v4_DRAM_layer_norm_weights_1, v5_DRAM_FF_weights_1, v6_DRAM_FF_weights_2, v7_DRAM_layer_norm_weights_2, v8_DRAM_output);
  printf("PASS smoke\n");
  return 0;
}
