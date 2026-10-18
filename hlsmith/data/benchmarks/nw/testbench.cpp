/*
 * Vitis HLS C-simulation testbench for nw (Needleman-Wunsch) benchmark.
 *
 * Strategy:
 *   1. Compute golden reference using the original needwun() algorithm
 *   2. Call the HLS workload() function with the same inputs
 *   3. Compare alignedA/alignedB outputs byte-by-byte
 *
 * Input sequences sourced from MachSuite/ML4Accel dataset.
 * Returns 0 on success, 1 on mismatch.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nw.h"

/* ── Scoring constants (must match the kernel) ──────────────────────── */
#define MATCH_SCORE 1
#define MISMATCH_SCORE -1
#define GAP_SCORE -1
#define ALIGN '\\'
#define SKIPA '^'
#define SKIPB '<'
#define MAX(A,B) ( ((A)>(B))?(A):(B) )

/* ── Golden reference: plain-C needwun (no HLS pragmas) ─────────────── */
static void needwun_ref(char SEQA[ALEN], char SEQB[BLEN],
                        char alignedA[ALEN+BLEN], char alignedB[ALEN+BLEN],
                        int M[(ALEN+1)*(BLEN+1)], char ptr[(ALEN+1)*(BLEN+1)]) {
    int score, up_left, up, left, mx;
    int row, row_up, r;
    int a_idx, b_idx;
    int a_str_idx, b_str_idx;

    for (a_idx = 0; a_idx < (ALEN+1); a_idx++)
        M[a_idx] = a_idx * GAP_SCORE;
    for (b_idx = 0; b_idx < (BLEN+1); b_idx++)
        M[b_idx*(ALEN+1)] = b_idx * GAP_SCORE;

    for (b_idx = 1; b_idx < (BLEN+1); b_idx++) {
        for (a_idx = 1; a_idx < (ALEN+1); a_idx++) {
            score = (SEQA[a_idx-1] == SEQB[b_idx-1]) ? MATCH_SCORE : MISMATCH_SCORE;
            row_up = (b_idx-1)*(ALEN+1);
            row    = b_idx*(ALEN+1);
            up_left = M[row_up + (a_idx-1)] + score;
            up      = M[row_up + a_idx]     + GAP_SCORE;
            left    = M[row   + (a_idx-1)]  + GAP_SCORE;
            mx = MAX(up_left, MAX(up, left));
            M[row + a_idx] = mx;
            if      (mx == left)    ptr[row + a_idx] = SKIPB;
            else if (mx == up)      ptr[row + a_idx] = SKIPA;
            else                    ptr[row + a_idx] = ALIGN;
        }
    }

    a_idx = ALEN; b_idx = BLEN;
    a_str_idx = 0; b_str_idx = 0;
    while (a_idx > 0 || b_idx > 0) {
        r = b_idx*(ALEN+1);
        if (ptr[r + a_idx] == ALIGN) {
            alignedA[a_str_idx++] = SEQA[a_idx-1];
            alignedB[b_str_idx++] = SEQB[b_idx-1];
            a_idx--; b_idx--;
        } else if (ptr[r + a_idx] == SKIPB) {
            alignedA[a_str_idx++] = SEQA[a_idx-1];
            alignedB[b_str_idx++] = '-';
            a_idx--;
        } else {
            alignedA[a_str_idx++] = '-';
            alignedB[b_str_idx++] = SEQB[b_idx-1];
            b_idx--;
        }
    }
    for (; a_str_idx < ALEN+BLEN; a_str_idx++) alignedA[a_str_idx] = '_';
    for (; b_str_idx < ALEN+BLEN; b_str_idx++) alignedB[b_str_idx] = '_';
}

/* ── Declaration of the HLS top function under test ────────────────────
 * Note: Vitis HLS csim_design compiles kernel + testbench together and
 * handles C/C++ linkage automatically. For standalone g++ testing,
 * the kernel's workload() must have matching linkage.
 */
extern "C" void workload(char* SEQA, char* SEQB,
              char* alignedA, char* alignedB, int num_jobs);

/* ── Test sequences (from MachSuite / ML4Accel dataset) ──────────────── */
static const char SEQ_A[] =
    "tcgacgaaataggatgacagcacgttctcgtattagagggccgcggtacaaaccaaatgctgcggcgtacagggcacggggcgctgttcgggagatcgggggaatcgtggcgtgggtgattcgccggc";
static const char SEQ_B[] =
    "ttcgagggcgcgtgtcgcggtccatcgacatgcccggtcggtgggacgtgggcgcctgatatagaggaatgcgattggaaggtcggacgggtcggcgagttgggcccggtgaatctgccatggtcgat";

int main() {
    int num_jobs = 1;
    int errors = 0;

    /* Allocate buffers */
    char seqA[ALEN];
    char seqB[BLEN];
    char ref_alignedA[ALEN+BLEN];
    char ref_alignedB[ALEN+BLEN];
    char dut_alignedA[ALEN+BLEN];
    char dut_alignedB[ALEN+BLEN];
    int  ref_M[(ALEN+1)*(BLEN+1)];
    char ref_ptr[(ALEN+1)*(BLEN+1)];

    /* Initialize inputs */
    memset(seqA, 0, sizeof(seqA));
    memset(seqB, 0, sizeof(seqB));
    memcpy(seqA, SEQ_A, ALEN);
    memcpy(seqB, SEQ_B, BLEN);

    /* Compute golden reference */
    memset(ref_alignedA, 0, sizeof(ref_alignedA));
    memset(ref_alignedB, 0, sizeof(ref_alignedB));
    memset(ref_M, 0, sizeof(ref_M));
    memset(ref_ptr, 0, sizeof(ref_ptr));
    needwun_ref(seqA, seqB, ref_alignedA, ref_alignedB, ref_M, ref_ptr);

    /* Call HLS design under test */
    memset(dut_alignedA, 0, sizeof(dut_alignedA));
    memset(dut_alignedB, 0, sizeof(dut_alignedB));
    workload(seqA, seqB, dut_alignedA, dut_alignedB, num_jobs);

    /* Compare outputs */
    if (memcmp(dut_alignedA, ref_alignedA, ALEN+BLEN) != 0) {
        printf("FAIL: alignedA mismatch\n");
        printf("  REF: %.40s...\n", ref_alignedA);
        printf("  DUT: %.40s...\n", dut_alignedA);
        errors++;
    }
    if (memcmp(dut_alignedB, ref_alignedB, ALEN+BLEN) != 0) {
        printf("FAIL: alignedB mismatch\n");
        printf("  REF: %.40s...\n", ref_alignedB);
        printf("  DUT: %.40s...\n", dut_alignedB);
        errors++;
    }

    if (errors == 0) {
        printf("PASS: nw testbench — all outputs match golden reference\n");
    } else {
        printf("FAIL: nw testbench — %d error(s)\n", errors);
    }

    return errors;
}
