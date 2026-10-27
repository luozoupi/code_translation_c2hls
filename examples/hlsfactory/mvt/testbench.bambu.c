/* Explicit MDPI argument extents; numeric inputs and original testbench unchanged.
 * This is a separate protocol variant. It does not disable golden comparison.
 */
#include "kernel.h"
#ifdef __BAMBU_SIM__
#include <mdpi/mdpi_user.h>
static void sized_top(double x1[N], double x2[N], double y_1[N], double y_2[N], double A[N][N]) {
  m_param_alloc(0, sizeof(double[N]));
  m_param_alloc(1, sizeof(double[N]));
  m_param_alloc(2, sizeof(double[N]));
  m_param_alloc(3, sizeof(double[N]));
  m_param_alloc(4, sizeof(double[N][N]));
  kernel_mvt(x1, x2, y_1, y_2, A);
}
#define kernel_mvt sized_top
#endif
#include "pilot_testbench.inc"
