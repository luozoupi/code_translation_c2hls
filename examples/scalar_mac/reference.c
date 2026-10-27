/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 luozoupi */
#include "kernel.h"

uint32_t scalar_mac(uint32_t x, uint32_t y, uint32_t bias)
{
    uint32_t product = 0;
    for (uint32_t index = 0; index < y; ++index) {
        product += x;
    }
    return product + bias;
}
