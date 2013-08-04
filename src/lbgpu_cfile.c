/* 
   Copyright (C) 2010,2011,2012,2013 The ESPResSo project

   This file is part of ESPResSo.
  
   ESPResSo is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   ESPResSo is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   
   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

/** \file lbgpu_cfile.c
 *
 * C file for the Lattice Boltzmann implementation on GPUs.
 * Header file for \ref lbgpu.h.
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "lbgpu.h"
#include "utils.h"
#include "communication.h"
#include "thermostat.h"
#include "grid.h"
#include "domain_decomposition.h"
#include "integrate.h"
#include "interaction_data.h"
#include "particle_data.h"
#include "global.h"
#include "lb-boundaries.h"
#include "cuda_common.h"
#ifdef LB_GPU


#ifndef D3Q19
#error The implementation only works for D3Q19 so far!
#endif

#if (LB_COMPONENTS==1) 
#    define  SC0 {0.0}
#    define  SC20 {0.0, 0.0}
#    define  SC1 {1.0}
#    define  SCM1 {-1.0}
#endif 
#if (LB_COMPONENTS==2) 
#    define  SC0 { 0.0 , 0.0 } 
#    define  SC20 {0.0, 0.0, 0.0, 0.0}
#    define  SC1 { 1.0 , 1.0 } 
#    define  SCM1 { -1.0, -1.0 } 
#endif 


/** Struct holding the Lattice Boltzmann parameters */
LB_parameters_gpu lbpar_gpu = { .rho=SC0, .mu=SC0, .viscosity=SC0, .gamma_shear=SC0, .gamma_bulk=SC0,
                                .gamma_odd=SC0,.gamma_even=SC0, .agrid=0.0, .tau=-1.0, .friction=SC0, .time_step=0.0, .lb_coupl_pref=SC1 ,
                                .lb_coupl_pref2=SC0, .bulk_viscosity=SCM1, .dim_x=0, .dim_y=0, .dim_z=0, .number_of_nodes=0, 
                                .number_of_particles=0, .fluct=0, .calc_val=1, .external_force=0, .ext_force={0.0, 0.0, 0.0}, 
                                 .your_seed=12345, .reinit=0,
#ifdef SHANCHEN
                                .coupling=SC20, .gamma_mobility=SC1
#endif
};
/**multi GPU para struct */
LB_gpus lbdevicepar_gpu;
/** this is the array that stores the hydrodynamic fields for the output */
LB_rho_v_pi_gpu *host_values = NULL;

LB_nodes_gpu *host_nodes = NULL;



/** Flag indicating momentum exchange between particles and fluid */
int transfer_momentum_gpu = 0;

static int max_ran = 1000000;
/*@}*/
//static double tau;

/** measures the MD time since the last fluid update */
static int fluidstep = 0;

/** c_sound_square in LB units*/
static float c_sound_sq = 1.f/3.f;

//clock_t start, end;
int i;


int n_extern_nodeforces = 0;
LB_extern_nodeforce_gpu *host_extern_nodeforces = NULL;

/*-----------------------------------------------------------*/
/** main of lb_gpu_programm */
/*-----------------------------------------------------------*/
#ifdef SHANCHEN
/* called from forces.c. This is at the beginning of the force
   calculation loop, so we increment the fluidstep counter here,
   and we reset it only when the last call to a LB function
   [lattice_boltzmann_update_gpu()] is performed within integrate_vv()
 */
void lattice_boltzmann_calc_shanchen_gpu(void){

  int factor = (int)round(lbpar_gpu.tau/time_step);

  if (fluidstep+1 >= factor) 
     lb_calc_shanchen_GPU();
}
#endif //SHANCHEN

/** lattice boltzmann update gpu called from integrate.c
*/

void lattice_boltzmann_update_gpu() {

  int factor = (int)round(lbpar_gpu.tau/time_step);

  fluidstep += 1;

  if (fluidstep>=factor) {

    fluidstep=0; 
    lb_integrate_GPU();
    LB_TRACE (fprintf(stderr,"lb_integrate_GPU \n"));

  }
}



/** (re-) allocation of the memory need for the particles (cpu part)*/
void lb_realloc_particles_gpu(){

  lbpar_gpu.number_of_particles = n_total_particles;
  LB_TRACE (printf("#particles realloc\t %u \n", lbpar_gpu.number_of_particles));
  //fprintf(stderr, "%u \t \n", lbpar_gpu.number_of_particles);
  /**-----------------------------------------------------*/
  /** allocating of the needed memory for several structs */
  /**-----------------------------------------------------*/

  lbpar_gpu.your_seed = (unsigned int)i_random(max_ran);

  LB_TRACE (fprintf(stderr,"test your_seed %u \n", lbpar_gpu.your_seed));

  lb_realloc_particle_GPU_leftovers(&lbpar_gpu);
}
/** (Re-)initializes the fluid according to the given value of rho. */
void lb_reinit_fluid_gpu() {

  //lbpar_gpu.your_seed = (unsigned int)i_random(max_ran);
  lb_reinit_parameters_gpu();
  if(lbpar_gpu.number_of_nodes != 0){
    lb_reinit_GPU(&lbpar_gpu, &lbdevicepar_gpu);
    lbpar_gpu.reinit = 1;
  }

  LB_TRACE (fprintf(stderr,"lb_reinit_fluid_gpu \n"));
}

/** Release the fluid. */
/*not needed in Espresso but still not deleted.
  Despite the name (TODO: change it), it releases 
  only the fluid-related memory on the gpu.*/
void lb_release_gpu(){

  if(host_nodes !=NULL) { free(host_nodes); host_nodes=NULL ;} 
  if(host_values!=NULL) { free(host_values); host_values=NULL;}
//  if(host_forces!=NULL) free(host_forces);
//  if(host_data  !=NULL) free(host_data);
}
/** (Re-)initializes the fluid. */
void lb_reinit_parameters_gpu() {
  int ii;

  lbpar_gpu.time_step = (float)time_step;
  for(ii=0;ii<LB_COMPONENTS;++ii){
    lbpar_gpu.mu[ii] = 0.0;
  
    if (lbpar_gpu.viscosity[ii] > 0.0) {
      /* Eq. (80) Duenweg, Schiller, Ladd, PRE 76(3):036704 (2007). */
      lbpar_gpu.gamma_shear[ii] = 1. - 2./(6.*lbpar_gpu.viscosity[ii]*lbpar_gpu.tau/(lbpar_gpu.agrid*lbpar_gpu.agrid) + 1.);   
    }
  
    if (lbpar_gpu.bulk_viscosity[ii] > 0.0) {
      /* Eq. (81) Duenweg, Schiller, Ladd, PRE 76(3):036704 (2007). */
      lbpar_gpu.gamma_bulk[ii] = 1. - 2./(9.*lbpar_gpu.bulk_viscosity[ii]*lbpar_gpu.tau/(lbpar_gpu.agrid*lbpar_gpu.agrid) + 1.);
    }
#ifdef SHANCHEN
    if (lbpar_gpu.mobility[0] > 0.0) {
      lbpar_gpu.gamma_mobility[0] = 1. - 2./(6.*lbpar_gpu.mobility[0]*lbpar_gpu.tau/(lbpar_gpu.agrid*lbpar_gpu.agrid) + 1.);
    }
#endif
    if (temperature > 0.0) {  /* fluctuating hydrodynamics ? */
  
      lbpar_gpu.fluct = 1;
  	LB_TRACE (fprintf(stderr, "fluct on \n"));
      /* Eq. (51) Duenweg, Schiller, Ladd, PRE 76(3):036704 (2007).*/
      /* Note that the modes are not normalized as in the paper here! */
  
      lbpar_gpu.mu[ii] = (float)temperature/c_sound_sq*lbpar_gpu.tau*lbpar_gpu.tau/(lbpar_gpu.agrid*lbpar_gpu.agrid); // TODO: check how to change mu
  
      /* lb_coupl_pref is stored in MD units (force)
       * Eq. (16) Ahlrichs and Duenweg, JCP 111(17):8225 (1999).
       * The factor 12 comes from the fact that we use random numbers
       * from -0.5 to 0.5 (equally distributed) which have variance 1/12.
       * time_step comes from the discretization.
       */
  
      lbpar_gpu.lb_coupl_pref[ii] = sqrt(12.f*2.f*lbpar_gpu.friction[ii]*(float)temperature/lbpar_gpu.time_step);
      lbpar_gpu.lb_coupl_pref2[ii] = sqrt(2.f*lbpar_gpu.friction[ii]*(float)temperature/lbpar_gpu.time_step);
  
    } else {
      /* no fluctuations at zero temperature */
      lbpar_gpu.fluct = 0;
      lbpar_gpu.lb_coupl_pref[ii] = 0.0;
      lbpar_gpu.lb_coupl_pref2[ii] = 0.0;
    }
  	LB_TRACE (fprintf(stderr,"lb_reinit_prarameters_gpu \n"));
  }
  reinit_parameters_GPU(&lbpar_gpu);
}

/** Performs a full initialization of
 *  the Lattice Boltzmann system. All derived parameters
 *  and the fluid are reset to their default values. */
void lb_init_gpu() {

  LB_TRACE(printf("Begin initialzing fluid on GPU\n"));

  /** set parameters for transfer to gpu */
  lb_reinit_parameters_gpu();

  lb_realloc_particles_gpu();

  lb_init_GPU(&lbpar_gpu, &lbdevicepar_gpu);
  
  gpu_init_particle_comm();
  cuda_bcast_global_part_params();

  LB_TRACE(printf("Initialzing fluid on GPU successful\n"));
}

/*@}*/


int lb_lbnode_set_extforce_GPU(int ind[3], double f[3])
{
  if ( ind[0] < 0 || ind[0] >=  lbpar_gpu.dim_x ||
       ind[1] < 0 || ind[1] >= lbpar_gpu.dim_y ||
       ind[2] < 0 || ind[2] >= lbpar_gpu.dim_z )
    return ES_ERROR;

  unsigned int index =
    ind[0] + ind[1]*lbpar_gpu.dim_x + ind[2]*lbpar_gpu.dim_x*lbpar_gpu.dim_y;

  size_t  size_of_extforces = (n_extern_nodeforces+1)*sizeof(LB_extern_nodeforce_gpu);
  host_extern_nodeforces = realloc(host_extern_nodeforces, size_of_extforces);
  
  host_extern_nodeforces[n_extern_nodeforces].force[0] = (float)f[0];
  host_extern_nodeforces[n_extern_nodeforces].force[1] = (float)f[1];
  host_extern_nodeforces[n_extern_nodeforces].force[2] = (float)f[2];
  
  host_extern_nodeforces[n_extern_nodeforces].index = index;
  n_extern_nodeforces++;
  
  if(lbpar_gpu.external_force == 0)lbpar_gpu.external_force = 1;

  lb_init_extern_nodeforces_GPU(n_extern_nodeforces, host_extern_nodeforces, &lbpar_gpu);

  return ES_OK;
}

#endif /* LB_GPU */
