/*
 * Strict Vitis HLS C-simulation testbench for cfd_step_factor (no testbench shipped originally).
 * Golden: plain.cpp's workload() copied below as workload_ref() and run on identical copies of deterministic inputs
 * (srand(42)). workload(result, variables, areas): result[SIZE] is a pure output (written before read) -> poisoned with
 * 12345.0f; variables, areas are inputs. Inputs are physically meaningful (density 0.8..1.4, small momentum, energy
 * such that pressure > 0, areas 0.5..1.5). Every result element checked, float tol 1e-3*fmax(1,|ref|), NaN = error.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cfd_step_factor.h"

extern "C" void workload(float result[SIZE], float variables[SIZE * NVAR], float areas[SIZE]);

/* ---- reference: copy of plain.cpp ---- */



static void compute_velocity_ref(float& density, float3& momentum, float3& velocity)
{
    velocity.x = momentum.x / density;
    velocity.y = momentum.y / density;
    velocity.z = momentum.z / density;
}

static float compute_speed_sqd_ref(float3& velocity)
{
    return velocity.x*velocity.x + velocity.y*velocity.y + velocity.z*velocity.z;
}

static float compute_pressure_ref(float& density, float& density_energy, float& speed_sqd)
{
    return (float(GAMMA) - float(1.0f))*(density_energy - float(0.5f)*density*speed_sqd);
}

static float compute_speed_of_sound_ref(float& density, float& pressure)
{
    return sqrt(float(GAMMA)*pressure / density);
}

static void workload_ref(float result[SIZE], float variables[SIZE * NVAR], float areas[SIZE])
{

    
    

    for (int i = 0; i < SIZE; i++)
    {
        float density = variables[NVAR*i + VAR_DENSITY];

        float3 momentum;
        momentum.x = variables[NVAR*i + (VAR_MOMENTUM + 0)];
        momentum.y = variables[NVAR*i + (VAR_MOMENTUM + 1)];
        momentum.z = variables[NVAR*i + (VAR_MOMENTUM + 2)];

        float density_energy = variables[NVAR*i + VAR_DENSITY_ENERGY];
        float3 velocity;       compute_velocity_ref(density, momentum, velocity);
        float speed_sqd      = compute_speed_sqd_ref(velocity);
        float pressure       = compute_pressure_ref(density, density_energy, speed_sqd);
        float speed_of_sound = compute_speed_of_sound_ref(density, pressure);

        result[i] = float(0.5f) / (sqrt(areas[i]) * (sqrt(speed_sqd) + speed_of_sound));
    }

    return;
}

/* ---- end reference ---- */

static int errors = 0;
static void chk_f(const char *name, int i, float d, float r) {
    if (!isfinite(r) || !(fabsf(d - r) <= 1e-3f * fmaxf(1.0f, fabsf(r)))) {   /* NaN/inf in DUT output (or a bad reference) is an error */
        if (errors < 10) printf("%s[%d]: got %g expected %g\n", name, i, (double)d, (double)r);
        errors++;
    }
}
static float frand(float lo, float hi) { return lo + (hi - lo) * ((float)rand() / (float)RAND_MAX); }

int main() {
    srand(42);
    static float variables[SIZE * NVAR], areas[SIZE], res_ref[SIZE], res_dut[SIZE];
    for (int i = 0; i < SIZE; i++) {
        variables[NVAR*i + VAR_DENSITY] = frand(0.8f, 1.4f);
        variables[NVAR*i + VAR_MOMENTUM + 0] = frand(0.0f, 1.5f);
        variables[NVAR*i + VAR_MOMENTUM + 1] = frand(-0.5f, 0.5f);
        variables[NVAR*i + VAR_MOMENTUM + 2] = frand(-0.5f, 0.5f);
        variables[NVAR*i + VAR_DENSITY_ENERGY] = frand(3.5f, 5.5f);
        areas[i] = frand(0.5f, 1.5f);
        res_ref[i] = 12345.0f;
        res_dut[i] = 12345.0f;
    }
    workload_ref(res_ref, variables, areas);
    workload(res_dut, variables, areas);
    for (int i = 0; i < SIZE; i++) chk_f("result", i, res_dut[i], res_ref[i]);
    if (errors) { printf("Failed with %d errors!\n", errors); return 1; }
    printf("Passed!\n");
    return 0;
}
