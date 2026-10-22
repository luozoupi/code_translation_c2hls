/* DPI-C arithmetic for the floating-point IP stand-ins written by rtl_prep.py.
 * IEEE-754 round-to-nearest-even, like AMD floating_point IP; subnormal operands and results are flushed to
 * (signed) zero, as the AMD cores do. Build with -ffp-contract=off and without -ffast-math so the compiler
 * keeps each operation a single correctly rounded IEEE op. */
#include <stdint.h>
#include <string.h>

static float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static double u2d(uint64_t u) { double d; memcpy(&d, &u, 8); return d; }
static uint64_t d2u(double d) { uint64_t u; memcpy(&u, &d, 8); return u; }

/* subnormal (exponent 0, mantissa != 0) -> signed zero */
static uint32_t ftz32(uint32_t u) { return ((u & 0x7f800000u) == 0) ? (u & 0x80000000u) : u; }
static uint64_t ftz64(uint64_t u) {
  return ((u & 0x7ff0000000000000ull) == 0) ? (u & 0x8000000000000000ull) : u;
}

#define FP32_OP(name, expr)                                        \
  uint32_t name(uint32_t a, uint32_t b) {                          \
    volatile float x = u2f(ftz32(a)), y = u2f(ftz32(b));           \
    volatile float r = (expr);                                     \
    return ftz32(f2u(r));                                          \
  }
#define FP64_OP(name, expr)                                        \
  uint64_t name(uint64_t a, uint64_t b) {                          \
    volatile double x = u2d(ftz64(a)), y = u2d(ftz64(b));          \
    volatile double r = (expr);                                    \
    return ftz64(d2u(r));                                          \
  }

FP32_OP(fp32_add, x + y)
FP32_OP(fp32_sub, x - y)
FP32_OP(fp32_mul, x * y)
FP64_OP(fp64_add, x + y)
FP64_OP(fp64_sub, x - y)
FP64_OP(fp64_mul, x * y)
