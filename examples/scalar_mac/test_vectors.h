/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 luozoupi */
#ifndef HLSMITH_EXAMPLE_SCALAR_MAC_TEST_VECTORS_H
#define HLSMITH_EXAMPLE_SCALAR_MAC_TEST_VECTORS_H

#include <stdint.h>

struct scalar_mac_case {
    uint32_t x;
    uint32_t y;
    uint32_t bias;
};

static const struct scalar_mac_case SCALAR_MAC_CASES[] = {
    {UINT32_C(0), UINT32_C(0), UINT32_C(0)},
    {UINT32_C(2), UINT32_C(3), UINT32_C(4)},
    {UINT32_C(7), UINT32_C(5), UINT32_C(1)},
    {UINT32_C(1), UINT32_C(0), UINT32_C(9)},
    {UINT32_C(0), UINT32_C(11), UINT32_C(123)},
    {UINT32_C(11), UINT32_C(1), UINT32_C(13)},
    {UINT32_C(9), UINT32_C(9), UINT32_C(7)},
    {UINT32_MAX, UINT32_C(2), UINT32_C(3)},
    {UINT32_C(0x80000000), UINT32_C(2), UINT32_C(1)},
    {UINT32_C(0x01020304), UINT32_C(4), UINT32_C(17)},
    {UINT32_MAX, UINT32_C(0), UINT32_C(0xabcdef01)},
    {UINT32_C(1234567), UINT32_C(15), UINT32_C(42)}
};

#define SCALAR_MAC_CASE_COUNT (sizeof(SCALAR_MAC_CASES) / sizeof(SCALAR_MAC_CASES[0]))

#endif
