/*
 * Strict Vitis HLS C-simulation testbench for AES-256 ECB.
 * Golden: plain.cpp (all AES functions) copied below with _ref suffix and run on identical copies of the inputs;
 * additionally the ciphertext is compared with the NIST SP 800-38A F.1.5 AES-256 ECB vector (known answer) and a
 * second deterministic (srand(42)) key/plaintext pair is checked against the reference.
 * workload(k, buf): k[32] is input only (plain.cpp copies it, never writes it back; the TB also checks it is unchanged),
 * buf[16] is in-out (plaintext in, ciphertext out; real data for both, no pure-output array exists here).
 * Integers: exact compare of every byte. Prints "Passed!" or "Failed with N errors!".
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "aes.h"

extern "C" void workload(uint8_t* k, uint8_t* buf);

/* ---- reference: copy of plain.cpp ---- */

#define F(x)   (((x)<<1) ^ ((((x)>>7) & 1) * 0x1b))
#define FD(x)  (((x) >> 1) ^ (((x) & 1) ? 0x8d : 0))

static const uint8_t sbox_ref[256] = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5,
    0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0,
    0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc,
    0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a,
    0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0,
    0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b,
    0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85,
    0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5,
    0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17,
    0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88,
    0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c,
    0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9,
    0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6,
    0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e,
    0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94,
    0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68,
    0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
};

#define rj_sbox(x)     sbox_ref[(x)]

static uint8_t rj_xtime_ref(uint8_t x) {
    return (x & 0x80) ? ((x << 1) ^ 0x1b) : (x << 1);
}

static void aes_subBytes_ref(uint8_t *buf) {
    uint8_t i = 16;
    sub: while (i--) buf[i] = rj_sbox(buf[i]);
}

static void aes_addRoundKey_ref(uint8_t *buf, uint8_t *key) {
    uint8_t i = 16;
    addkey: while (i--) buf[i] ^= key[i];
}

static void aes_addRoundKey_cpy_ref(uint8_t *buf, uint8_t *key, uint8_t *cpk) {
    uint8_t i = 16;
    cpkey: while (i--) buf[i] ^= (cpk[i] = key[i]), cpk[16+i] = key[16 + i];
}

static void aes_shiftRows_ref(uint8_t *buf) {
    uint8_t i, j;
    i = buf[1]; buf[1] = buf[5]; buf[5] = buf[9]; buf[9] = buf[13]; buf[13] = i;
    i = buf[10]; buf[10] = buf[2]; buf[2] = i;
    j = buf[3]; buf[3] = buf[15]; buf[15] = buf[11]; buf[11] = buf[7]; buf[7] = j;
    j = buf[14]; buf[14] = buf[6]; buf[6] = j;
}

static void aes_mixColumns_ref(uint8_t *buf) {
    uint8_t i, a, b, c, d, e;
    mix: for (i = 0; i < 16; i += 4) {
        a = buf[i]; b = buf[i + 1]; c = buf[i + 2]; d = buf[i + 3];
        e = a ^ b ^ c ^ d;
        buf[i] ^= e ^ rj_xtime_ref(a^b);     buf[i+1] ^= e ^ rj_xtime_ref(b^c);
        buf[i+2] ^= e ^ rj_xtime_ref(c^d);   buf[i+3] ^= e ^ rj_xtime_ref(d^a);
    }
}

static void aes_expandEncKey_ref(uint8_t *k, uint8_t *rc) {
    uint8_t i;
    k[0] ^= rj_sbox(k[29]) ^ (*rc);
    k[1] ^= rj_sbox(k[30]);
    k[2] ^= rj_sbox(k[31]);
    k[3] ^= rj_sbox(k[28]);
    *rc = F(*rc);
    exp1: for (i = 4; i < 16; i += 4)
        k[i] ^= k[i-4], k[i+1] ^= k[i-3], k[i+2] ^= k[i-2], k[i+3] ^= k[i-1];
    k[16] ^= rj_sbox(k[12]);
    k[17] ^= rj_sbox(k[13]);
    k[18] ^= rj_sbox(k[14]);
    k[19] ^= rj_sbox(k[15]);
    exp2: for (i = 20; i < 32; i += 4)
        k[i] ^= k[i-4], k[i+1] ^= k[i-3], k[i+2] ^= k[i-2], k[i+3] ^= k[i-1];
}

static void aes256_encrypt_ecb_ref(aes256_context *ctx, uint8_t k[32], uint8_t buf[16]) {
    uint8_t rcon = 1;
    uint8_t i;

    ecb1: for (i = 0; i < sizeof(ctx->key); i++) {
        ctx->enckey[i] = ctx->deckey[i] = k[i];
    }
    ecb2: for (i = 8; --i; ) {
        aes_expandEncKey_ref(ctx->deckey, &rcon);
    }

    aes_addRoundKey_cpy_ref(buf, ctx->enckey, ctx->key);
    ecb3: for (i = 1, rcon = 1; i < 14; ++i) {
        aes_subBytes_ref(buf);
        aes_shiftRows_ref(buf);
        aes_mixColumns_ref(buf);
        if (i & 1) aes_addRoundKey_ref(buf, &ctx->key[16]);
        else aes_expandEncKey_ref(ctx->key, &rcon), aes_addRoundKey_ref(buf, ctx->key);
    }
    aes_subBytes_ref(buf);
    aes_shiftRows_ref(buf);
    aes_expandEncKey_ref(ctx->key, &rcon);
    aes_addRoundKey_ref(buf, ctx->key);
}


static void workload_ref(uint8_t* k, uint8_t* buf) {

    uint8_t l_k[32], l_buf[16];
    int i;
    for (i = 0; i < 32; i++) l_k[i] = k[i];
    for (i = 0; i < 16; i++) l_buf[i] = buf[i];

    aes256_context ctx;
    aes256_encrypt_ecb_ref(&ctx, l_k, l_buf);

    for (i = 0; i < 16; i++) buf[i] = l_buf[i];
}
/* ---- end reference ---- */

static int errors = 0;
static void chk_i(const char *name, long i, long long d, long long r) {
    if (d != r) {
        if (errors < 10) printf("%s[%ld]: got %lld expected %lld\n", name, i, d, r);
        errors++;
    }
}

static void run_case(const char *tag, const uint8_t *key, const uint8_t *pt, const uint8_t *known /* may be NULL */) {
    uint8_t ref_buf[16], dut_buf[16], ref_key[32], dut_key[32];
    memcpy(ref_key, key, 32); memcpy(ref_buf, pt, 16);
    memcpy(dut_key, key, 32); memcpy(dut_buf, pt, 16);
    workload_ref(ref_key, ref_buf);
    workload(dut_key, dut_buf);
    for (int i = 0; i < 16; i++) {
        chk_i(tag, i, dut_buf[i], ref_buf[i]);
        if (known) chk_i("nist", i, dut_buf[i], known[i]);
    }
    for (int i = 0; i < 32; i++) chk_i("key", i, dut_key[i], key[i]);
}

int main() {
    /* Test vector from MachSuite input.data / NIST SP 800-38A F.1.5 */
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
    uint8_t nist[16] = {
        0xf3, 0xee, 0xd1, 0xbd, 0xb5, 0xd2, 0xa0, 0x3c,
        0x06, 0x4b, 0x5a, 0x7e, 0x3d, 0xb1, 0x81, 0xf8
    };
    run_case("ct_nist", key, plaintext, nist);

    srand(42);
    uint8_t key2[32], pt2[16];
    for (int i = 0; i < 32; i++) key2[i] = (uint8_t)(rand() & 0xff);
    for (int i = 0; i < 16; i++) pt2[i] = (uint8_t)(rand() & 0xff);
    run_case("ct_rand", key2, pt2, NULL);
    if (errors) { printf("Failed with %d errors!\n", errors); return 1; }
    printf("Passed!\n");
    return 0;
}
