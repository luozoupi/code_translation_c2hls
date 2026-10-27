/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 luozoupi */
#include "kernel.h"

void array_sum(const uint32_t input[4], uint32_t output[1])
{
    output[0] = input[0] + input[1] + input[2] + input[3];
}
