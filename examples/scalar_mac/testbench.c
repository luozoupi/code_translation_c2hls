/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 luozoupi */
#include <inttypes.h>
#include <stddef.h>
#include <stdio.h>

#include "kernel.h"
#include "test_vectors.h"

int main(void)
{
    printf("HLSMITH_RESULT");
    for (size_t index = 0; index < SCALAR_MAC_CASE_COUNT; ++index) {
        const struct scalar_mac_case input = SCALAR_MAC_CASES[index];
        printf(" %" PRIu32, scalar_mac(input.x, input.y, input.bias));
    }
    printf("\n");
    return 0;
}
