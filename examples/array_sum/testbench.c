/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 luozoupi */
#include <inttypes.h>
#include <stdio.h>

#include "kernel.h"

int main(void)
{
    const uint32_t input[4] = {UINT32_C(1), UINT32_C(2), UINT32_C(3), UINT32_C(4)};
    uint32_t output[1] = {0};
    array_sum(input, output);
    printf("HLSMITH_RESULT %" PRIu32 "\n", output[0]);
    return 0;
}
