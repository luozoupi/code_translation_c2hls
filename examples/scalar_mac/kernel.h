/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 luozoupi */
#ifndef HLSMITH_EXAMPLE_SCALAR_MAC_H
#define HLSMITH_EXAMPLE_SCALAR_MAC_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Unsigned arithmetic: the result is (x * y + bias) modulo 2^32. */
uint32_t scalar_mac(uint32_t x, uint32_t y, uint32_t bias);

#ifdef __cplusplus
}
#endif

#endif
