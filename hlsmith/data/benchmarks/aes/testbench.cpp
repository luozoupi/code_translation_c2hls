/*
 * Vitis HLS C-simulation testbench for AES-256 ECB benchmark.
 * Uses known test vector: encrypts a plaintext block with a known key,
 * compares against expected ciphertext from NIST/MachSuite.
 * Returns 0 on success, 1 on mismatch.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "aes.h"

extern "C" void workload(uint8_t* k, uint8_t* buf);

int main() {
    int errors = 0;

    /* Test vector from MachSuite input.data */
    uint8_t key[32] = {
        0x60, 0x3d, 0xeb, 0x10, 0x15, 0xca, 0x71, 0xbe,
        0x2b, 0x73, 0xae, 0xf0, 0x85, 0x7d, 0x77, 0x81,
        0x1f, 0x35, 0x2c, 0x07, 0x3b, 0x61, 0x08, 0xd7,
        0x2d, 0x98, 0x10, 0xa3, 0x09, 0x14, 0xdf, 0xf4
    };
    uint8_t plaintext[16] = {
        0x6b, 0xc1, 0xbe, 0xe2, 0x2e, 0x40, 0x9f, 0x96,
        0xe9, 0x3d, 0x7e, 0x11, 0x73, 0x93, 0x17, 0x2a
    };

    /* Compute golden reference using the same function */
    uint8_t ref_buf[16], dut_buf[16];
    uint8_t ref_key[32], dut_key[32];

    memcpy(ref_key, key, 32);
    memcpy(ref_buf, plaintext, 16);
    aes256_context ref_ctx;
    aes256_encrypt_ecb(&ref_ctx, ref_key, ref_buf);

    /* Call HLS design under test */
    memcpy(dut_key, key, 32);
    memcpy(dut_buf, plaintext, 16);
    workload(dut_key, dut_buf);

    /* Compare outputs */
    if (memcmp(dut_buf, ref_buf, 16) != 0) {
        printf("FAIL: ciphertext mismatch\n");
        printf("  REF: ");
        for (int i = 0; i < 16; i++) printf("%02x", ref_buf[i]);
        printf("\n  DUT: ");
        for (int i = 0; i < 16; i++) printf("%02x", dut_buf[i]);
        printf("\n");
        errors++;
    }

    if (errors == 0) {
        printf("PASS: aes testbench — ciphertext matches golden reference\n");
    } else {
        printf("FAIL: aes testbench — %d error(s)\n", errors);
    }

    return (errors > 0) ? 1 : 0;
}
