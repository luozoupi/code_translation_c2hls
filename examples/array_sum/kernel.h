/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 luozoupi */
#ifndef HLSMITH_EXAMPLE_ARRAY_SUM_H
#define HLSMITH_EXAMPLE_ARRAY_SUM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Store the sum of four input words, modulo 2^32, in output[0]. */
void array_sum(const uint32_t input[4], uint32_t output[1]);

#ifdef __cplusplus
}
#endif

#endif
