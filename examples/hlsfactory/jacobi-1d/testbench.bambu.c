/* Explicit MDPI argument extents; numeric inputs and original testbench unchanged.
 * This is a separate protocol variant. It does not disable golden comparison.
 */
#include "kernel.h"
#ifdef __BAMBU_SIM__
#include <mdpi/mdpi_user.h>
static void sized_top(double A[N], double B[N]) {
  m_param_alloc(0, sizeof(double[N]));
  m_param_alloc(1, sizeof(double[N]));
  kernel_jacobi_1d(A, B);
}
#define kernel_jacobi_1d sized_top
#endif
#include "pilot_testbench.inc"
