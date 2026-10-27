/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 luozoupi */
#include "kernel.h"

extern "C" void array_sum(const uint32_t input[4], uint32_t output[1])
{
#ifdef __SYNTHESIS__
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0 depth=4
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem1 depth=1
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
#endif
    output[0] = input[0] + input[1] + input[2] + input[3];
}
