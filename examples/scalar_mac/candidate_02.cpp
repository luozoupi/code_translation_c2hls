/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 luozoupi */
#include "kernel.h"

extern "C" uint32_t scalar_mac(uint32_t x, uint32_t y, uint32_t bias)
{
    const uint64_t product = static_cast<uint64_t>(x) * static_cast<uint64_t>(y);
    return static_cast<uint32_t>(product + static_cast<uint64_t>(bias));
}
