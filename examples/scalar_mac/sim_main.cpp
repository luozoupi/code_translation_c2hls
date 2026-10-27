/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 luozoupi */
#include <cinttypes>
#include <cstddef>
#include <cstdio>

#include "Vdut.h"
#include "verilated.h"
#include "test_vectors.h"

static void tick(Vdut &dut)
{
    dut.clk = 0;
    dut.eval();
    dut.clk = 1;
    dut.eval();
}

int main(int argc, char **argv)
{
    Verilated::commandArgs(argc, argv);
    Vdut dut;
    dut.clk = 0;
    dut.rst_n = 0;
    dut.start = 0;
    dut.x = 0;
    dut.y = 0;
    dut.bias = 0;
    tick(dut);
    dut.rst_n = 1;

    uint32_t outputs[SCALAR_MAC_CASE_COUNT] = {};
    uint64_t cycles = 0;
    for (std::size_t index = 0; index < SCALAR_MAC_CASE_COUNT; ++index) {
        const scalar_mac_case input = SCALAR_MAC_CASES[index];
        dut.x = input.x;
        dut.y = input.y;
        dut.bias = input.bias;
        dut.start = 1;
        unsigned wait_cycles = 0;
        do {
            tick(dut);
            ++cycles;
            ++wait_cycles;
        } while (!dut.done && wait_cycles < 8 && !Verilated::gotFinish());
        dut.start = 0;
        if (!dut.done || Verilated::gotFinish()) {
            std::fprintf(stderr, "transaction %zu did not finish within eight cycles\n", index);
            dut.final();
            return 2;
        }
        outputs[index] = dut.result;
    }

    std::printf("HLSMITH_RESULT");
    for (std::size_t index = 0; index < SCALAR_MAC_CASE_COUNT; ++index)
        std::printf(" %" PRIu32, outputs[index]);
    std::printf("\nHLSMITH_CYCLES:%" PRIu64 "\n", cycles);
    dut.final();
    return 0;
}
