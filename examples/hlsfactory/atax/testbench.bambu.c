/* Explicit MDPI argument extents; numeric inputs and original testbench unchanged.
 * This is a separate protocol variant. It does not disable golden comparison.
 */
#include "kernel.h"
#ifdef __BAMBU_SIM__
#include <mdpi/mdpi_user.h>
static void sized_top(double A[M][N], double x[N], double y[N], double tmp[M]) {
  m_param_alloc(0, sizeof(double[M][N]));
  m_param_alloc(1, sizeof(double[N]));
  m_param_alloc(2, sizeof(double[N]));
  m_param_alloc(3, sizeof(double[M]));
  kernel_atax(A, x, y, tmp);
}
#define kernel_atax sized_top
#endif
#include "pilot_testbench.inc"
