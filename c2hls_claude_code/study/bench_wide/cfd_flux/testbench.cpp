/*
 * Strict Vitis HLS C-simulation testbench for cfd_flux (no testbench shipped originally).
 * Golden: plain.cpp's cfd_flux()/workload() copied below as *_ref and run on identical copies of deterministic inputs
 * (srand(42)). workload(result, elements_surrounding_elements, normals, variables, fc_momentum_x/y/z, fc_density_energy):
 * result[SIZE*NVAR] is a pure output (every element written) -> poisoned with 12345.0f; all other args are inputs only.
 * Inputs are physically meaningful: density 0.8..1.4, small momentum, energy such that pressure > 0, flux
 * contributions fc_* computed from the state (as in Rodinia CFD), neighbours: random element, with ~10% wall (-1) and
 * ~10% far-field (-2) faces. Every result element checked, float tol 1e-3*fmax(1,|ref|), NaN = error.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cfd_flux.h"

extern "C" void workload(float result[SIZE * NVAR], float elements_surrounding_elements [SIZE * NNB],
                         float normals [SIZE * NNB * NDIM], float variables [SIZE * NVAR],
                         float fc_momentum_x [SIZE * NDIM], float fc_momentum_y [SIZE * NDIM],
                         float fc_momentum_z [SIZE * NDIM], float fc_density_energy [SIZE * NDIM]);

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



static void cfd_flux_ref(float result[SIZE * NVAR], float elements_surrounding_elements [SIZE * NNB], float normals [SIZE * NNB * NDIM], float variables [SIZE * NVAR], float fc_momentum_x [SIZE * NDIM], float fc_momentum_y [SIZE * NDIM], float fc_momentum_z [SIZE * NDIM], float fc_density_energy [SIZE * NDIM])
{


    const float smoothing_coefficient = float(0.2f);

    float ff_variable[NVAR] = {1.399999976158142090, 1.680000066757202148, 0.000000000000000000, 0.000000000000000000, 3.507999897003173828};

    float3 ff_fc_momentum_x, ff_fc_momentum_y,ff_fc_momentum_z, ff_fc_density_energy;

    ff_fc_momentum_x.x = 3.016000270843505859; ff_fc_momentum_x.y = 0.000000000000000000; ff_fc_momentum_x.z = 0.000000000000000000; 
    ff_fc_momentum_y.x = 0.000000000000000000; ff_fc_momentum_y.y = 1.000000000000000000; ff_fc_momentum_y.z = 0.000000000000000000; 
    ff_fc_momentum_z.x = 0.000000000000000000; ff_fc_momentum_z.y = 0.000000000000000000; ff_fc_momentum_z.z = 1.000000000000000000; 
    ff_fc_density_energy.x = 5.409600257873535156; ff_fc_density_energy.y = 0.000000000000000000; ff_fc_density_energy.z = 0.000000000000000000;


    for(int i = 0; i < SIZE; i++)
    {
        int j, nb;
        float3 normal; float normal_len;
        float factor;

        float density_i = variables[NVAR*i + VAR_DENSITY];
        float3 momentum_i;
        momentum_i.x = variables[NVAR*i + (VAR_MOMENTUM+0)];
        momentum_i.y = variables[NVAR*i + (VAR_MOMENTUM+1)];
        momentum_i.z = variables[NVAR*i + (VAR_MOMENTUM+2)];
        float density_energy_i = variables[NVAR*i + VAR_DENSITY_ENERGY];

        float3 velocity_i;                           compute_velocity_ref(density_i, momentum_i, velocity_i);
        float speed_sqd_i                          = compute_speed_sqd_ref(velocity_i);
        float speed_i                              = sqrt(speed_sqd_i);
        float pressure_i                           = compute_pressure_ref(density_i, density_energy_i, speed_sqd_i);
        float speed_of_sound_i                     = compute_speed_of_sound_ref(density_i, pressure_i);
        float3 fc_i_momentum_x, fc_i_momentum_y, fc_i_momentum_z;
        float3 fc_i_density_energy;

        fc_i_momentum_x.x = fc_momentum_x[i*NDIM + 0];
        fc_i_momentum_x.y = fc_momentum_x[i*NDIM + 1];
        fc_i_momentum_x.z = fc_momentum_x[i*NDIM + 2];

        fc_i_momentum_y.x = fc_momentum_y[i*NDIM + 0];
        fc_i_momentum_y.y = fc_momentum_y[i*NDIM + 1];
        fc_i_momentum_y.z = fc_momentum_y[i*NDIM + 2];

        fc_i_momentum_z.x = fc_momentum_z[i*NDIM + 0];
        fc_i_momentum_z.y = fc_momentum_z[i*NDIM + 1];
        fc_i_momentum_z.z = fc_momentum_z[i*NDIM + 2];

        fc_i_density_energy.x = fc_density_energy[i*NDIM + 0];
        fc_i_density_energy.y = fc_density_energy[i*NDIM + 1];
        fc_i_density_energy.z = fc_density_energy[i*NDIM + 2];

        float flux_i_density = float(0.0f);
        float3 flux_i_momentum;
        flux_i_momentum.x = float(0.0f);
        flux_i_momentum.y = float(0.0f);
        flux_i_momentum.z = float(0.0f);
        float flux_i_density_energy = float(0.0f);

        float3 velocity_nb;
        float density_nb, density_energy_nb;
        float3 momentum_nb;
        float3 fc_nb_momentum_x, fc_nb_momentum_y, fc_nb_momentum_z;
        float3 fc_nb_density_energy;
        float speed_sqd_nb, speed_of_sound_nb, pressure_nb;

        for(j = 0; j < NNB; j++)
        {
            nb = elements_surrounding_elements[i*NNB + j];
            normal.x = normals[(i*NNB + j)*NDIM + 0];
            normal.y = normals[(i*NNB + j)*NDIM + 1];
            normal.z = normals[(i*NNB + j)*NDIM + 2];
            normal_len = sqrt(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z);

            if(nb >= 0)
            {
                density_nb =        variables[nb*NVAR + VAR_DENSITY];
                momentum_nb.x =     variables[nb*NVAR + (VAR_MOMENTUM+0)];
                momentum_nb.y =     variables[nb*NVAR + (VAR_MOMENTUM+1)];
                momentum_nb.z =     variables[nb*NVAR + (VAR_MOMENTUM+2)];
                density_energy_nb = variables[nb*NVAR + VAR_DENSITY_ENERGY];
                                                    compute_velocity_ref(density_nb, momentum_nb, velocity_nb);
                speed_sqd_nb                      = compute_speed_sqd_ref(velocity_nb);
                pressure_nb                       = compute_pressure_ref(density_nb, density_energy_nb, speed_sqd_nb);
                speed_of_sound_nb                 = compute_speed_of_sound_ref(density_nb, pressure_nb);
                fc_nb_momentum_x.x = fc_momentum_x[nb*NDIM + 0];
                fc_nb_momentum_x.y = fc_momentum_x[nb*NDIM + 1];
                fc_nb_momentum_x.z = fc_momentum_x[nb*NDIM + 2];

                fc_nb_momentum_y.x = fc_momentum_y[nb*NDIM + 0];
                fc_nb_momentum_y.y = fc_momentum_y[nb*NDIM + 1];
                fc_nb_momentum_y.z = fc_momentum_y[nb*NDIM + 2];


                fc_nb_momentum_z.x = fc_momentum_z[nb*NDIM + 0];
                fc_nb_momentum_z.y = fc_momentum_z[nb*NDIM + 1];
                fc_nb_momentum_z.z = fc_momentum_z[nb*NDIM + 2];

                fc_nb_density_energy.x = fc_density_energy[nb*NDIM + 0];
                fc_nb_density_energy.y = fc_density_energy[nb*NDIM + 1];
                fc_nb_density_energy.z = fc_density_energy[nb*NDIM + 2];

                factor = -normal_len*smoothing_coefficient*float(0.5f)*(speed_i + sqrt(speed_sqd_nb) + speed_of_sound_i + speed_of_sound_nb);
                flux_i_density += factor*(density_i-density_nb);
                flux_i_density_energy += factor*(density_energy_i-density_energy_nb);
                flux_i_momentum.x += factor*(momentum_i.x-momentum_nb.x);
                flux_i_momentum.y += factor*(momentum_i.y-momentum_nb.y);
                flux_i_momentum.z += factor*(momentum_i.z-momentum_nb.z);

                factor = float(0.5f)*normal.x;
                flux_i_density += factor*(momentum_nb.x+momentum_i.x);
                flux_i_density_energy += factor*(fc_nb_density_energy.x+fc_i_density_energy.x);
                flux_i_momentum.x += factor*(fc_nb_momentum_x.x+fc_i_momentum_x.x);
                flux_i_momentum.y += factor*(fc_nb_momentum_y.x+fc_i_momentum_y.x);
                flux_i_momentum.z += factor*(fc_nb_momentum_z.x+fc_i_momentum_z.x);

                factor = float(0.5f)*normal.y;
                flux_i_density += factor*(momentum_nb.y+momentum_i.y);
                flux_i_density_energy += factor*(fc_nb_density_energy.y+fc_i_density_energy.y);
                flux_i_momentum.x += factor*(fc_nb_momentum_x.y+fc_i_momentum_x.y);
                flux_i_momentum.y += factor*(fc_nb_momentum_y.y+fc_i_momentum_y.y);
                flux_i_momentum.z += factor*(fc_nb_momentum_z.y+fc_i_momentum_z.y);

                factor = float(0.5f)*normal.z;
                flux_i_density += factor*(momentum_nb.z+momentum_i.z);
                flux_i_density_energy += factor*(fc_nb_density_energy.z+fc_i_density_energy.z);
                flux_i_momentum.x += factor*(fc_nb_momentum_x.z+fc_i_momentum_x.z);
                flux_i_momentum.y += factor*(fc_nb_momentum_y.z+fc_i_momentum_y.z);
                flux_i_momentum.z += factor*(fc_nb_momentum_z.z+fc_i_momentum_z.z);
            }
            else if(nb == -1)
            {
                flux_i_momentum.x += normal.x*pressure_i;
                flux_i_momentum.y += normal.y*pressure_i;
                flux_i_momentum.z += normal.z*pressure_i;
            }
            else if(nb == -2)
            {
                factor = float(0.5f)*normal.x;
                flux_i_density += factor*(ff_variable[VAR_MOMENTUM+0]+momentum_i.x);
                flux_i_density_energy += factor*(ff_fc_density_energy.x+fc_i_density_energy.x);
                flux_i_momentum.x += factor*(ff_fc_momentum_x.x + fc_i_momentum_x.x);
                flux_i_momentum.y += factor*(ff_fc_momentum_y.x + fc_i_momentum_y.x);
                flux_i_momentum.z += factor*(ff_fc_momentum_z.x + fc_i_momentum_z.x);

                factor = float(0.5f)*normal.y;
                flux_i_density += factor*(ff_variable[VAR_MOMENTUM+1]+momentum_i.y);
                flux_i_density_energy += factor*(ff_fc_density_energy.y+fc_i_density_energy.y);
                flux_i_momentum.x += factor*(ff_fc_momentum_x.y + fc_i_momentum_x.y);
                flux_i_momentum.y += factor*(ff_fc_momentum_y.y + fc_i_momentum_y.y);
                flux_i_momentum.z += factor*(ff_fc_momentum_z.y + fc_i_momentum_z.y);

                factor = float(0.5f)*normal.z;
                flux_i_density += factor*(ff_variable[VAR_MOMENTUM+2]+momentum_i.z);
                flux_i_density_energy += factor*(ff_fc_density_energy.z+fc_i_density_energy.z);
                flux_i_momentum.x += factor*(ff_fc_momentum_x.z + fc_i_momentum_x.z);
                flux_i_momentum.y += factor*(ff_fc_momentum_y.z + fc_i_momentum_y.z);
                flux_i_momentum.z += factor*(ff_fc_momentum_z.z + fc_i_momentum_z.z);

            }
        }

        result[i*NVAR + VAR_DENSITY] = flux_i_density;
        result[i*NVAR + (VAR_MOMENTUM+0)] = flux_i_momentum.x;
        result[i*NVAR + (VAR_MOMENTUM+1)] = flux_i_momentum.y;
        result[i*NVAR + (VAR_MOMENTUM+2)] = flux_i_momentum.z;
        result[i*NVAR + VAR_DENSITY_ENERGY] = flux_i_density_energy;
    }

    return;
}

static void workload_ref(float result[SIZE * NVAR], float elements_surrounding_elements [SIZE * NNB], float normals [SIZE * NNB * NDIM], 
                float variables [SIZE * NVAR], float fc_momentum_x [SIZE * NDIM], float fc_momentum_y [SIZE * NDIM], 
                float fc_momentum_z [SIZE * NDIM], float fc_density_energy [SIZE * NDIM])
{

    




    cfd_flux_ref(result, elements_surrounding_elements, normals, variables, fc_momentum_x, fc_momentum_y, fc_momentum_z, fc_density_energy);
    

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

static float res_ref[SIZE * NVAR], res_dut[SIZE * NVAR];
static float esel[SIZE * NNB], normals[SIZE * NNB * NDIM], variables[SIZE * NVAR];
static float fcx[SIZE * NDIM], fcy[SIZE * NDIM], fcz[SIZE * NDIM], fcde[SIZE * NDIM];

int main() {
    srand(42);
    for (int i = 0; i < SIZE; i++) {
        float rho = frand(0.8f, 1.4f);
        float m[3] = { frand(0.0f, 1.5f), frand(-0.5f, 0.5f), frand(-0.5f, 0.5f) };
        float E = frand(3.5f, 5.5f);
        variables[NVAR*i + VAR_DENSITY] = rho;
        for (int d = 0; d < 3; d++) variables[NVAR*i + VAR_MOMENTUM + d] = m[d];
        variables[NVAR*i + VAR_DENSITY_ENERGY] = E;
        float v[3] = { m[0]/rho, m[1]/rho, m[2]/rho };
        float sq = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
        float p = (1.4f - 1.0f) * (E - 0.5f * rho * sq);
        /* fc_momentum_x/y/z[d] = m_row * v[d] + p*(row==d); fc_density_energy[d] = v[d]*(E+p) */
        for (int d = 0; d < 3; d++) {
            fcx[i*NDIM + d] = m[0] * v[d] + (d == 0 ? p : 0.0f);
            fcy[i*NDIM + d] = m[1] * v[d] + (d == 1 ? p : 0.0f);
            fcz[i*NDIM + d] = m[2] * v[d] + (d == 2 ? p : 0.0f);
            fcde[i*NDIM + d] = v[d] * (E + p);
        }
        for (int j = 0; j < NNB; j++) {
            int r = rand() % 10;
            esel[i*NNB + j] = (r == 0) ? -1.0f : (r == 1) ? -2.0f : (float)(rand() % SIZE);
            for (int d = 0; d < NDIM; d++) normals[(i*NNB + j)*NDIM + d] = frand(-1.0f, 1.0f);
        }
    }
    for (int i = 0; i < SIZE * NVAR; i++) { res_ref[i] = 12345.0f; res_dut[i] = 12345.0f; }

    workload_ref(res_ref, esel, normals, variables, fcx, fcy, fcz, fcde);
    workload(res_dut, esel, normals, variables, fcx, fcy, fcz, fcde);
    for (int i = 0; i < SIZE * NVAR; i++) chk_f("result", i, res_dut[i], res_ref[i]);
    if (errors) { printf("Failed with %d errors!\n", errors); return 1; }
    printf("Passed!\n");
    return 0;
}
