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

/** \file lbgpu.cu
 *
 * Cuda (.cu) file for the Lattice Boltzmann implementation on GPUs.
 * Header file for \ref lbgpu.h.
 */

#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <mpi.h>

#include "config.hpp" 
#include "lbgpu.hpp"
#include "cuda.hpp"
#include "communication.hpp"
#include "grid.hpp"

#ifndef GAUSSRANDOM
#define GAUSSRANDOM
#endif
#define EXTERNAL_FORCES

/** measures the MD time since the last fluid update 
    This is duplicated from lbgpu_cfile.c because of SHANCHEN force
    update is different from LB, and we need to avoid calculating/adding
    forces on the fluid when only those on particle matter. Maybe
    one can find a better solution.
**/ 
/**defining structures residing in global memory */
/** struct for phys. values */
#if 0
static LB_values_gpu *device_values = NULL;
/** structs for velocity densities */
static LB_nodes_gpu nodes_a;
static LB_nodes_gpu nodes_b;
/** struct for particle force */
static LB_particle_force_gpu *particle_force = NULL;
/** struct for particle position and veloctiy */
static LB_particle_gpu *particle_data = NULL;
/** struct for node force */
static LB_node_force_gpu node_f;
/** struct for storing particle rn seed */
static LB_particle_seed_gpu *part = NULL;

static LB_extern_nodeforce_gpu *extern_nodeforces = NULL;

static float* lb_boundary_force = NULL;
static float* lb_boundary_velocity = NULL;
#endif
plan_gpu *plan;
#ifdef LB_BOUNDARIES_GPU
/** pointer for bound index array*/
static int *boundary_node_list;
static int *boundary_index_list;
static __device__ __constant__ int n_lb_boundaries_gpu = 0;
static size_t size_of_boundindex;
#endif
/** number of gpus **/
static int gpu_n = 0;

/**defining size values for allocating global memory */
static size_t size_of_values;
static size_t size_of_values_wo_halo;
static size_t size_of_forces;
static size_t size_of_positions;
static size_t size_of_seed;
static size_t size_of_extern_nodeforces;
static size_t size_of_uint;
static size_t size_of_nodes_gpu;
static size_t size_of_3floats; 
static size_t size_of_buffer[3];
/**parameters residing in constant memory */
static __device__ __constant__ LB_parameters_gpu para;
static __device__ __constant__ LB_gpus devpara;
static const float c_sound_sq = 1.f/3.f;
/**cudasteams for parallel computing on cpu and gpu */
cudaStream_t *stream;

cudaError_t err;
cudaError_t _err;

int plan_initflag = 0;
/*********************************************************/
/** \name device funktions called by kernel funktions */
/*********************************************************/
/*-------------------------------------------------------*/

/*-------------------------------------------------------*/

/** atomic add function for sveral cuda architectures
*/
__device__ inline void atomicadd(float* address, float value){
#if !defined __CUDA_ARCH__ || __CUDA_ARCH__ >= 200 // for Fermi, atomicAdd supports floats
  atomicAdd(address, value);
#elif __CUDA_ARCH__ >= 110
#warning Using slower atomicAdd emulation
// float-atomic-add from
// [url="http://forums.nvidia.com/index.php?showtopic=158039&view=findpost&p=991561"]
  float old = value;
  while ((old = atomicExch(address, atomicExch(address, 0.0f)+old))!=0.0f);
#else
#error I need at least compute capability 1.1
#endif
}

/**randomgenerator which generates numbers [0,1]
 * @param *rn	Pointer to randomnumber array of the local node or particle
*/
__device__ void random_01(LB_randomnr_gpu *rn){

  const float mxi = 1.f/(float)(1ul<<31);
  unsigned int state = rn->seed;

  state = 1103515245 * state + 12345;
  rn->randomnr[0] = (float)(state & ((1ul<<31)-1))*mxi;
  state = 1103515245 * state + 12345;
  rn->randomnr[1] = (float)(state & ((1ul<<31)-1))*mxi;
  rn->seed = state;

}
/**randomgenerator which generates numbers [0,1]
 * @param *rn	Pointer to randomnumber array of the local node or particle
*/
__device__ void random_minstd(LB_randomnr_gpu *rn){

  unsigned int state = rn->seed;
  const int mxi = (1u << 31) - 1, a = 16807;
  state = ((long int)state)*a % mxi;
  rn->randomnr[0] = (float)((state-1)/((1u << 31) - 1));
  state = ((long int)state)*a % mxi;
  rn->randomnr[1] = (float)((state-1)/((1u << 31) - 1));
  rn->seed = state;
  
}

/** gaussian random nummber generator for thermalisation
 * @param *rn	Pointer to randomnumber array of the local node node or particle
*/
__device__ void gaussian_random(LB_randomnr_gpu *rn){

  float x1, x2;
  float r2, fac;
  /** On every second call two gaussian random numbers are calculated
   via the Box-Muller transformation.*/
  /** draw two uniform random numbers in the unit circle */
  do {
    random_01(rn);
    x1 = 2.f*rn->randomnr[0]-1.f;
    x2 = 2.f*rn->randomnr[1]-1.f;
    r2 = x1*x1 + x2*x2;
  } while (r2 >= 1.f || r2 == 0.f);

  /** perform Box-Muller transformation */
  fac = sqrtf(-2.f*__logf(r2)/r2);
  rn->randomnr[0] = x2*fac;
  rn->randomnr[1] = x1*fac;

}

/**tranformation from 1d array-index to xyz
 * @param index		node index / thread index (Input)
 * @param xyz		Pointer to calculated xyz array (Output)
 */
__device__ void index_to_xyz(unsigned int index, unsigned int *xyz){

  xyz[0] = index%para.dim_x;
  index /= para.dim_x;
  xyz[1] = index%para.dim_y;
  index /= para.dim_y;
  xyz[2] = index;
}

#ifndef SHANCHEN


/**calculation of the modes from the velocitydensities (space-transform.)
 * @param n_a		Pointer to local node residing in array a (Input)
 * @param index		node index / thread index (Input)
 * @param mode		Pointer to the local register values mode (Output)
*/
__device__ void calc_m_from_n(LB_nodes_gpu n_a, unsigned int index, float *mode){

  /* mass mode */
  mode[0] = n_a.vd[0*para.number_of_nodes + index] + n_a.vd[1*para.number_of_nodes + index] + n_a.vd[2*para.number_of_nodes + index]
          + n_a.vd[3*para.number_of_nodes + index] + n_a.vd[4*para.number_of_nodes + index] + n_a.vd[5*para.number_of_nodes + index]
          + n_a.vd[6*para.number_of_nodes + index] + n_a.vd[7*para.number_of_nodes + index] + n_a.vd[8*para.number_of_nodes + index]
          + n_a.vd[9*para.number_of_nodes + index] + n_a.vd[10*para.number_of_nodes + index] + n_a.vd[11*para.number_of_nodes + index] + n_a.vd[12*para.number_of_nodes + index]
          + n_a.vd[13*para.number_of_nodes + index] + n_a.vd[14*para.number_of_nodes + index] + n_a.vd[15*para.number_of_nodes + index] + n_a.vd[16*para.number_of_nodes + index]
          + n_a.vd[17*para.number_of_nodes + index] + n_a.vd[18*para.number_of_nodes + index];

  /* momentum modes */
  mode[1] = (n_a.vd[1*para.number_of_nodes + index] - n_a.vd[2*para.number_of_nodes + index]) + (n_a.vd[7*para.number_of_nodes + index] - n_a.vd[8*para.number_of_nodes + index])
          + (n_a.vd[9*para.number_of_nodes + index] - n_a.vd[10*para.number_of_nodes + index]) + (n_a.vd[11*para.number_of_nodes + index] - n_a.vd[12*para.number_of_nodes + index])
          + (n_a.vd[13*para.number_of_nodes + index] - n_a.vd[14*para.number_of_nodes + index]);
  mode[2] = (n_a.vd[3*para.number_of_nodes + index] - n_a.vd[4*para.number_of_nodes + index]) + (n_a.vd[7*para.number_of_nodes + index] - n_a.vd[8*para.number_of_nodes + index])
          - (n_a.vd[9*para.number_of_nodes + index] - n_a.vd[10*para.number_of_nodes + index]) + (n_a.vd[15*para.number_of_nodes + index] - n_a.vd[16*para.number_of_nodes + index])
          + (n_a.vd[17*para.number_of_nodes + index] - n_a.vd[18*para.number_of_nodes + index]);
  mode[3] = (n_a.vd[5*para.number_of_nodes + index] - n_a.vd[6*para.number_of_nodes + index]) + (n_a.vd[11*para.number_of_nodes + index] - n_a.vd[12*para.number_of_nodes + index])
          - (n_a.vd[13*para.number_of_nodes + index] - n_a.vd[14*para.number_of_nodes + index]) + (n_a.vd[15*para.number_of_nodes + index] - n_a.vd[16*para.number_of_nodes + index])
          - (n_a.vd[17*para.number_of_nodes + index] - n_a.vd[18*para.number_of_nodes + index]);
//printf("mode 1: %lf mode 2: %lf mode 3: %lf\n", mode[1], mode[2], mode[3]);
  /* stress modes */
  mode[4] = -(n_a.vd[0*para.number_of_nodes + index]) + n_a.vd[7*para.number_of_nodes + index] + n_a.vd[8*para.number_of_nodes + index] + n_a.vd[9*para.number_of_nodes + index] + n_a.vd[10*para.number_of_nodes + index]
          + n_a.vd[11*para.number_of_nodes + index] + n_a.vd[12*para.number_of_nodes + index] + n_a.vd[13*para.number_of_nodes + index] + n_a.vd[14*para.number_of_nodes + index]
          + n_a.vd[15*para.number_of_nodes + index] + n_a.vd[16*para.number_of_nodes + index] + n_a.vd[17*para.number_of_nodes + index] + n_a.vd[18*para.number_of_nodes + index];
  mode[5] = n_a.vd[1*para.number_of_nodes + index] + n_a.vd[2*para.number_of_nodes + index] - (n_a.vd[3*para.number_of_nodes + index] + n_a.vd[4*para.number_of_nodes + index])
          + (n_a.vd[11*para.number_of_nodes + index] + n_a.vd[12*para.number_of_nodes + index]) + (n_a.vd[13*para.number_of_nodes + index] + n_a.vd[14*para.number_of_nodes + index])
          - (n_a.vd[15*para.number_of_nodes + index] + n_a.vd[16*para.number_of_nodes + index]) - (n_a.vd[17*para.number_of_nodes + index] + n_a.vd[18*para.number_of_nodes + index]);
  mode[6] = (n_a.vd[1*para.number_of_nodes + index] + n_a.vd[2*para.number_of_nodes + index]) + (n_a.vd[3*para.number_of_nodes + index] + n_a.vd[4*para.number_of_nodes + index])
          - (n_a.vd[11*para.number_of_nodes + index] + n_a.vd[12*para.number_of_nodes + index]) - (n_a.vd[13*para.number_of_nodes + index] + n_a.vd[14*para.number_of_nodes + index])
          - (n_a.vd[15*para.number_of_nodes + index] + n_a.vd[16*para.number_of_nodes + index]) - (n_a.vd[17*para.number_of_nodes + index] + n_a.vd[18*para.number_of_nodes + index])
          - 2.f*(n_a.vd[5*para.number_of_nodes + index] + n_a.vd[6*para.number_of_nodes + index] - (n_a.vd[7*para.number_of_nodes + index] + n_a.vd[8*para.number_of_nodes + index])
          - (n_a.vd[9*para.number_of_nodes + index] +n_a.vd[10*para.number_of_nodes + index]));
  mode[7] = n_a.vd[7*para.number_of_nodes + index] + n_a.vd[8*para.number_of_nodes + index] - (n_a.vd[9*para.number_of_nodes + index] + n_a.vd[10*para.number_of_nodes + index]);
  mode[8] = n_a.vd[11*para.number_of_nodes + index] + n_a.vd[12*para.number_of_nodes + index] - (n_a.vd[13*para.number_of_nodes + index] + n_a.vd[14*para.number_of_nodes + index]);
  mode[9] = n_a.vd[15*para.number_of_nodes + index] + n_a.vd[16*para.number_of_nodes + index] - (n_a.vd[17*para.number_of_nodes + index] + n_a.vd[18*para.number_of_nodes + index]);

//printf("mode 4: %lf mode 5: %lf mode 6: %lf\n", mode[4], mode[5], mode[6]);
  /* kinetic modes */
  mode[10] = -2.f*(n_a.vd[1*para.number_of_nodes + index] - n_a.vd[2*para.number_of_nodes + index]) + (n_a.vd[7*para.number_of_nodes + index] - n_a.vd[8*para.number_of_nodes + index])
           + (n_a.vd[9*para.number_of_nodes + index] - n_a.vd[10*para.number_of_nodes + index]) + (n_a.vd[11*para.number_of_nodes + index] - n_a.vd[12*para.number_of_nodes + index])
           + (n_a.vd[13*para.number_of_nodes + index] - n_a.vd[14*para.number_of_nodes + index]);
  mode[11] = -2.f*(n_a.vd[3*para.number_of_nodes + index] - n_a.vd[4*para.number_of_nodes + index]) + (n_a.vd[7*para.number_of_nodes + index] - n_a.vd[8*para.number_of_nodes + index])
           - (n_a.vd[9*para.number_of_nodes + index] - n_a.vd[10*para.number_of_nodes + index]) + (n_a.vd[15*para.number_of_nodes + index] - n_a.vd[16*para.number_of_nodes + index])
           + (n_a.vd[17*para.number_of_nodes + index] - n_a.vd[18*para.number_of_nodes + index]);
  mode[12] = -2.f*(n_a.vd[5*para.number_of_nodes + index] - n_a.vd[6*para.number_of_nodes + index]) + (n_a.vd[11*para.number_of_nodes + index] - n_a.vd[12*para.number_of_nodes + index])
           - (n_a.vd[13*para.number_of_nodes + index] - n_a.vd[14*para.number_of_nodes + index]) + (n_a.vd[15*para.number_of_nodes + index] - n_a.vd[16*para.number_of_nodes + index])
           - (n_a.vd[17*para.number_of_nodes + index] - n_a.vd[18*para.number_of_nodes + index]);
  mode[13] = (n_a.vd[7*para.number_of_nodes + index] - n_a.vd[8*para.number_of_nodes + index]) + (n_a.vd[9*para.number_of_nodes + index] - n_a.vd[10*para.number_of_nodes + index])
           - (n_a.vd[11*para.number_of_nodes + index] - n_a.vd[12*para.number_of_nodes + index]) - (n_a.vd[13*para.number_of_nodes + index] - n_a.vd[14*para.number_of_nodes + index]);
  mode[14] = (n_a.vd[7*para.number_of_nodes + index] - n_a.vd[8*para.number_of_nodes + index]) - (n_a.vd[9*para.number_of_nodes + index] - n_a.vd[10*para.number_of_nodes + index])
           - (n_a.vd[15*para.number_of_nodes + index] - n_a.vd[16*para.number_of_nodes + index]) - (n_a.vd[17*para.number_of_nodes + index] - n_a.vd[18*para.number_of_nodes + index]);
  mode[15] = (n_a.vd[11*para.number_of_nodes + index] - n_a.vd[12*para.number_of_nodes + index]) - (n_a.vd[13*para.number_of_nodes + index] - n_a.vd[14*para.number_of_nodes + index])
           - (n_a.vd[15*para.number_of_nodes + index] - n_a.vd[16*para.number_of_nodes + index]) + (n_a.vd[17*para.number_of_nodes + index] - n_a.vd[18*para.number_of_nodes + index]);
  mode[16] = n_a.vd[0*para.number_of_nodes + index] + n_a.vd[7*para.number_of_nodes + index] + n_a.vd[8*para.number_of_nodes + index] + n_a.vd[9*para.number_of_nodes + index] + n_a.vd[10*para.number_of_nodes + index]
           + n_a.vd[11*para.number_of_nodes + index] + n_a.vd[12*para.number_of_nodes + index] + n_a.vd[13*para.number_of_nodes + index] + n_a.vd[14*para.number_of_nodes + index]
           + n_a.vd[15*para.number_of_nodes + index] + n_a.vd[16*para.number_of_nodes + index] + n_a.vd[17*para.number_of_nodes + index] + n_a.vd[18*para.number_of_nodes + index]
           - 2.f*((n_a.vd[1*para.number_of_nodes + index] + n_a.vd[2*para.number_of_nodes + index]) + (n_a.vd[3*para.number_of_nodes + index] + n_a.vd[4*para.number_of_nodes + index])
           + (n_a.vd[5*para.number_of_nodes + index] + n_a.vd[6*para.number_of_nodes + index]));
  mode[17] = -(n_a.vd[1*para.number_of_nodes + index] + n_a.vd[2*para.number_of_nodes + index]) + (n_a.vd[3*para.number_of_nodes + index] + n_a.vd[4*para.number_of_nodes + index])
           + (n_a.vd[11*para.number_of_nodes + index] + n_a.vd[12*para.number_of_nodes + index]) + (n_a.vd[13*para.number_of_nodes + index] + n_a.vd[14*para.number_of_nodes + index])
           - (n_a.vd[15*para.number_of_nodes + index] + n_a.vd[16*para.number_of_nodes + index]) - (n_a.vd[17*para.number_of_nodes + index] + n_a.vd[18*para.number_of_nodes + index]);
  mode[18] = -(n_a.vd[1*para.number_of_nodes + index] + n_a.vd[2*para.number_of_nodes + index]) - (n_a.vd[3*para.number_of_nodes + index] + n_a.vd[4*para.number_of_nodes + index])
           - (n_a.vd[11*para.number_of_nodes + index] + n_a.vd[12*para.number_of_nodes + index]) - (n_a.vd[13*para.number_of_nodes + index] + n_a.vd[14*para.number_of_nodes + index])
           - (n_a.vd[15*para.number_of_nodes + index] + n_a.vd[16*para.number_of_nodes + index]) - (n_a.vd[17*para.number_of_nodes + index] + n_a.vd[18*para.number_of_nodes + index])
           + 2.f*((n_a.vd[5*para.number_of_nodes + index] + n_a.vd[6*para.number_of_nodes + index]) + (n_a.vd[7*para.number_of_nodes + index] + n_a.vd[8*para.number_of_nodes + index])
           + (n_a.vd[9*para.number_of_nodes + index] + n_a.vd[10*para.number_of_nodes + index]));

}

/**lb_relax_modes, means collision update of the modes
 * @param index		node index / thread index (Input)
 * @param mode		Pointer to the local register values mode (Input/Output)
 * @param node_f	Pointer to local node force (Input)
*/
__device__ void relax_modes(float *mode, unsigned int index, LB_node_force_gpu node_f){

  float Rho = mode[0] + para.rho*para.agrid*para.agrid*para.agrid;
  float j[3], pi_eq[6];

  /** re-construct the real density
  * remember that the populations are stored as differences to their
  * equilibrium value */

  j[0] = mode[1];
  j[1] = mode[2];
  j[2] = mode[3];

  /** if forces are present, the momentum density is redefined to
  * inlcude one half-step of the force action.  See the
  * Chapman-Enskog expansion in [Ladd & Verberg]. */

  j[0] += 0.5f*node_f.force[0*para.number_of_nodes + index];
  j[1] += 0.5f*node_f.force[1*para.number_of_nodes + index];
  j[2] += 0.5f*node_f.force[2*para.number_of_nodes + index];

  /** equilibrium part of the stress modes (eq13 schiller)*/
  pi_eq[0] = ((j[0]*j[0])+(j[1]*j[1])+(j[2]*j[2]))/Rho;
  pi_eq[1] = ((j[0]*j[0])-(j[1]*j[1]))/Rho;
  pi_eq[2] = (((j[0]*j[0])+(j[1]*j[1])+(j[2]*j[2])) - 3.0f*(j[2]*j[2]))/Rho;
  pi_eq[3] = j[0]*j[1]/Rho;
  pi_eq[4] = j[0]*j[2]/Rho;
  pi_eq[5] = j[1]*j[2]/Rho;

  /** relax the stress modes (eq14 schiller)*/
  mode[4] = pi_eq[0] + para.gamma_bulk*(mode[4] - pi_eq[0]);
  mode[5] = pi_eq[1] + para.gamma_shear*(mode[5] - pi_eq[1]);
  mode[6] = pi_eq[2] + para.gamma_shear*(mode[6] - pi_eq[2]);
  mode[7] = pi_eq[3] + para.gamma_shear*(mode[7] - pi_eq[3]);
  mode[8] = pi_eq[4] + para.gamma_shear*(mode[8] - pi_eq[4]);
  mode[9] = pi_eq[5] + para.gamma_shear*(mode[9] - pi_eq[5]);

  /** relax the ghost modes (project them out) */
  /** ghost modes have no equilibrium part due to orthogonality */
  mode[10] = para.gamma_odd*mode[10];
  mode[11] = para.gamma_odd*mode[11];
  mode[12] = para.gamma_odd*mode[12];
  mode[13] = para.gamma_odd*mode[13];
  mode[14] = para.gamma_odd*mode[14];
  mode[15] = para.gamma_odd*mode[15];
  mode[16] = para.gamma_even*mode[16];
  mode[17] = para.gamma_even*mode[17];
  mode[18] = para.gamma_even*mode[18];

}

/**thermalization of the modes with gaussian random numbers
 * @param index		node index / thread index (Input)
 * @param mode		Pointer to the local register values mode (Input/Output)
 * @param *rn		Pointer to randomnumber array of the local node
*/
__device__ void thermalize_modes(float *mode, unsigned int index, LB_randomnr_gpu *rn){

  float Rho = mode[0] + para.rho*para.agrid*para.agrid*para.agrid;

  /*
    if (Rho <0)
    printf("Rho too small! %f %f %f", Rho, mode[0], para.rho*para.agrid*para.agrid*para.agrid);
  */
#ifdef GAUSSRANDOM
  /** stress modes */
  gaussian_random(rn);
  mode[4] += sqrt(Rho*(para.mu*(2.f/3.f)*(1.f-(para.gamma_bulk*para.gamma_bulk)))) * rn->randomnr[1];
  mode[5] += sqrt(Rho*(para.mu*(4.f/9.f)*(1.f-(para.gamma_shear*para.gamma_shear)))) * rn->randomnr[0];

  gaussian_random(rn);
  mode[6] += sqrt(Rho*(para.mu*(4.f/3.f)*(1.f-(para.gamma_shear*para.gamma_shear)))) * rn->randomnr[1];
  mode[7] += sqrt(Rho*(para.mu*(1.f/9.f)*(1.f-(para.gamma_shear*para.gamma_shear)))) * rn->randomnr[0];

  gaussian_random(rn);
  mode[8] += sqrt(Rho*(para.mu*(1.f/9.f)*(1.f-(para.gamma_shear*para.gamma_shear)))) * rn->randomnr[1];
  mode[9] += sqrt(Rho*(para.mu*(1.f/9.f)*(1.f-(para.gamma_shear*para.gamma_shear)))) * rn->randomnr[0];

  /** ghost modes */
  gaussian_random(rn);
  mode[10] += sqrt(Rho*(para.mu*(2.f/3.f))) * rn->randomnr[1];
  mode[11] += sqrt(Rho*(para.mu*(2.f/3.f))) * rn->randomnr[0];

  gaussian_random(rn);
  mode[12] += sqrt(Rho*(para.mu*(2.f/3.f))) * rn->randomnr[1];
  mode[13] += sqrt(Rho*(para.mu*(2.f/9.f))) * rn->randomnr[0];

  gaussian_random(rn);
  mode[14] += sqrt(Rho*(para.mu*(2.f/9.f))) * rn->randomnr[1];
  mode[15] += sqrt(Rho*(para.mu*(2.f/9.f))) * rn->randomnr[0];

  gaussian_random(rn);
  mode[16] += sqrt(Rho*(para.mu*(2.f))) * rn->randomnr[1];
  mode[17] += sqrt(Rho*(para.mu*(4.f/9.f))) * rn->randomnr[0];

  gaussian_random(rn);
  mode[18] += sqrt(Rho*(para.mu*(4.f/3.f))) * rn->randomnr[1];
#else
  /** stress modes */
  random_01(rn);
  mode[4] += sqrt(12.f*Rho*para.mu*(2.f/3.f)*(1.f-(para.gamma_bulk*para.gamma_bulk))) * (rn->randomnr[1]-0.5f);
  mode[5] += sqrt(12.f*Rho*para.mu*(4.f/9.f)*(1.f-(para.gamma_shear*para.gamma_shear))) * (rn->randomnr[0]-0.5f);

  random_01(rn);
  mode[6] += sqrt(12.f*Rho*para.mu*(4.f/3.f)*(1.f-(para.gamma_shear*para.gamma_shear))) * (rn->randomnr[1]-0.5f);
  mode[7] += sqrt(12.f*Rho*para.mu*(1.f/9.f)*(1.f-(para.gamma_shear*para.gamma_shear))) * (rn->randomnr[0]-0.5f);

  random_01(rn);
  mode[8] += sqrt(12.f*para.mu*(1.f/9.f)*(1.f-(para.gamma_shear*para.gamma_shear))) * (rn->randomnr[1]-0.5f);
  mode[9] += sqrt(12.f*para.mu*(1.f/9.f)*(1.f-(para.gamma_shear*para.gamma_shear))) * (rn->randomnr[0]-0.5f);

  /** ghost modes */
  random_01(rn);
  mode[10] += sqrt(12.f*Rho*para.mu*(2.f/3.f)) * (rn->randomnr[1]-0.5f);
  mode[11] += sqrt(12.f*Rho*para.mu*(2.f/3.f)) * (rn->randomnr[0]-0.5f);

  random_01(rn);
  mode[12] += sqrt(12.f*Rho*para.mu*(2.f/3.f)) * (rn->randomnr[1]-0.5f);
  mode[13] += sqrt(12.f*Rho*para.mu*(2.f/9.f)) * (rn->randomnr[0]-0.5f);

  random_01(rn);
  mode[14] += sqrt(12.f*Rho*para.mu*(2.f/9.f)) * (rn->randomnr[1]-0.5f);
  mode[15] += sqrt(12.f*Rho*para.mu*(2.f/9.f)) * (rn->randomnr[0]-0.5f);

  random_01(rn);
  mode[16] += sqrt(12.f*Rho*para.mu*(2.f)) * (rn->randomnr[1]-0.5f);
  mode[17] += sqrt(12.f*Rho*para.mu*(4.f/9.f)) * (rn->randomnr[0]-0.5f);

  random_01(rn);
  mode[18] += sqrt(12.f*Rho*para.mu*(4.f/3.f)) * (rn->randomnr[1]-0.5f);
#endif
}
/*-------------------------------------------------------*/
/**normalization of the modes need befor backtransformation into velocity space
 * @param mode		Pointer to the local register values mode (Input/Output)
*/
__device__ void normalize_modes(float* mode){

  /** normalization factors enter in the back transformation */
  mode[0] *= 1.f;
  mode[1] *= 3.f;
  mode[2] *= 3.f;
  mode[3] *= 3.f;
  mode[4] *= 3.f/2.f;
  mode[5] *= 9.f/4.f;
  mode[6] *= 3.f/4.f;
  mode[7] *= 9.f;
  mode[8] *= 9.f;
  mode[9] *= 9.f;
  mode[10] *= 3.f/2.f;
  mode[11] *= 3.f/2.f;
  mode[12] *= 3.f/2.f;
  mode[13] *= 9.f/2.f;
  mode[14] *= 9.f/2.f;
  mode[15] *= 9.f/2.f;
  mode[16] *= 1.f/2.f;
  mode[17] *= 9.f/4.f;
  mode[18] *= 3.f/4.f;

}
/*-------------------------------------------------------*/
/**backtransformation from modespace to desityspace and streaming with the push method using !!! pbc !!!
 * @param index		node index / thread index (Input)
 * @param mode		Pointer to the local register values mode (Input)
 * @param *n_b		Pointer to local node residing in array b (Output)
*/
__device__ void calc_n_from_modes_push(LB_nodes_gpu n_b, float *mode, unsigned int index){

  unsigned int xyz[3];
  index_to_xyz(index, xyz);
  unsigned int x = xyz[0];
  unsigned int y = xyz[1];
  unsigned int z = xyz[2];

  n_b.vd[0*para.number_of_nodes + x + para.dim_x*y + para.dim_x*para.dim_y*z] = 1.f/3.f * (mode[0] - mode[4] + mode[16]);
  n_b.vd[1*para.number_of_nodes + (x+1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*z] = 1.f/18.f * (mode[0] + mode[1] + mode[5] + mode[6] - mode[17] - mode[18] - 2.f*(mode[10] + mode[16]));
  n_b.vd[2*para.number_of_nodes + (para.dim_x+x-1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*z] = 1.f/18.f * (mode[0] - mode[1] + mode[5] + mode[6] - mode[17] - mode[18] + 2.f*(mode[10] - mode[16]));
  n_b.vd[3*para.number_of_nodes + x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*z] = 1.f/18.f * (mode[0] + mode[2] - mode[5] + mode[6] + mode[17] - mode[18] - 2.f*(mode[11] + mode[16]));
  n_b.vd[4*para.number_of_nodes + x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*z] = 1.f/18.f * (mode[0] - mode[2] - mode[5] + mode[6] + mode[17] - mode[18] + 2.f*(mode[11] - mode[16]));
  n_b.vd[5*para.number_of_nodes + x + para.dim_x*y + para.dim_x*para.dim_y*((z+1)%para.dim_z)] = 1.f/18.f * (mode[0] + mode[3] - 2.f*(mode[6] + mode[12] + mode[16] - mode[18]));
  n_b.vd[6*para.number_of_nodes + x + para.dim_x*y + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z)] = 1.f/18.f * (mode[0] - mode[3] - 2.f*(mode[6] - mode[12] + mode[16] - mode[18]));
  n_b.vd[7*para.number_of_nodes + (x+1)%para.dim_x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*z] = 1.f/36.f * (mode[0] + mode[1] + mode[2] + mode[4] + 2.f*mode[6] + mode[7] + mode[10] + mode[11] + mode[13] + mode[14] + mode[16] + 2.f*mode[18]);
  n_b.vd[8*para.number_of_nodes + (para.dim_x+x-1)%para.dim_x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*z] = 1.f/36.f * (mode[0] - mode[1] - mode[2] + mode[4] + 2.f*mode[6] + mode[7] - mode[10] - mode[11] - mode[13] - mode[14] + mode[16] + 2.f*mode[18]);
  n_b.vd[9*para.number_of_nodes + (x+1)%para.dim_x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*z] = 1.f/36.f * (mode[0] + mode[1] - mode[2] + mode[4] + 2.f*mode[6] - mode[7] + mode[10] - mode[11] + mode[13] - mode[14] + mode[16] + 2.f*mode[18]);
  n_b.vd[10*para.number_of_nodes + (para.dim_x+x-1)%para.dim_x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*z] = 1.f/36.f * (mode[0] - mode[1] + mode[2] + mode[4] + 2.f*mode[6] - mode[7] - mode[10] + mode[11] - mode[13] + mode[14] + mode[16] + 2.f*mode[18]);
  n_b.vd[11*para.number_of_nodes + (x+1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*((z+1)%para.dim_z)] = 1.f/36.f * (mode[0] + mode[1] + mode[3] + mode[4] + mode[5] - mode[6] + mode[8] + mode[10] + mode[12] - mode[13] + mode[15] + mode[16] + mode[17] - mode[18]);
  n_b.vd[12*para.number_of_nodes + (para.dim_x+x-1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z)] = 1.f/36.f * (mode[0] - mode[1] - mode[3] + mode[4] + mode[5] - mode[6] + mode[8] - mode[10] - mode[12] + mode[13] - mode[15] + mode[16] + mode[17] - mode[18]);
  n_b.vd[13*para.number_of_nodes + (x+1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z)] = 1.f/36.f * (mode[0] + mode[1] - mode[3] + mode[4] + mode[5] - mode[6] - mode[8] + mode[10] - mode[12] - mode[13] - mode[15] + mode[16] + mode[17] - mode[18]);
  n_b.vd[14*para.number_of_nodes + (para.dim_x+x-1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*((z+1)%para.dim_z)] = 1.f/36.f * (mode[0] - mode[1] + mode[3] + mode[4] + mode[5] - mode[6] - mode[8] - mode[10] + mode[12] + mode[13] + mode[15] + mode[16] + mode[17] - mode[18]);
  n_b.vd[15*para.number_of_nodes + x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*((z+1)%para.dim_z)] = 1.f/36.f * (mode[0] + mode[2] + mode[3] + mode[4] - mode[5] - mode[6] + mode[9] + mode[11] + mode[12] - mode[14] - mode[15] + mode[16] - mode[17] - mode[18]);
  n_b.vd[16*para.number_of_nodes + x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z)] = 1.f/36.f * (mode[0] - mode[2] - mode[3] + mode[4] - mode[5] - mode[6] + mode[9] - mode[11] - mode[12] + mode[14] + mode[15] + mode[16] - mode[17] - mode[18]);
  n_b.vd[17*para.number_of_nodes + x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z)] = 1.f/36.f * (mode[0] + mode[2] - mode[3] + mode[4] - mode[5] - mode[6] - mode[9] + mode[11] - mode[12] - mode[14] + mode[15] + mode[16] - mode[17] - mode[18]);
  n_b.vd[18*para.number_of_nodes + x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*((z+1)%para.dim_z)] = 1.f/36.f * (mode[0] - mode[2] + mode[3] + mode[4] - mode[5] - mode[6] - mode[9] - mode[11] + mode[12] + mode[14] - mode[15] + mode[16] - mode[17] - mode[18]);

}

/*-------------------------------------------------------*/
/**backtransformation from modespace to desityspace and streaming with the push method and buffering of border desities
 * @param index		node index / thread index (Input)
 * @param mode		Pointer to the local register values mode (Input)
 * @param *n_b		Pointer to local node residing in array b (Output)
 * @param *buffer Pointer to buffer (Output)
*/
__device__ void calc_n_from_modes_buffer(LB_nodes_gpu n_b, float *buffer, float *mode, unsigned int index){

  unsigned int xyz[3];
  index_to_xyz(index, xyz);
  unsigned int x = xyz[0];
  unsigned int y = xyz[1];
  unsigned int z = xyz[2];
  //printf("x %i, y %i, z %i\n", x,y,z);
  unsigned nhyz = para.number_of_halo_nodes[0];
  unsigned nhxz = para.number_of_halo_nodes[1];
  unsigned nhxy = para.number_of_halo_nodes[2];
  //right buffered n's
  if(x == para.dim_x-1){
    //vd1
    buffer[0*nhyz + y + (para.dim_y*z)] = 1.f/18.f * (mode[0] + mode[1] + mode[5] + mode[6] - mode[17] - mode[18] - 2.f*(mode[10] + mode[16]));
    //vd7
    buffer[1*nhyz + y + (para.dim_y*z)] = 1.f/36.f * (mode[0] + mode[1] + mode[2] + mode[4] + 2.f*mode[6] + mode[7] + mode[10] + mode[11] + mode[13] + mode[14] + mode[16] + 2.f*mode[18]);
    //vd9
    buffer[2*nhyz + y + (para.dim_y*z)] = 1.f/36.f * (mode[0] + mode[1] - mode[2] + mode[4] + 2.f*mode[6] - mode[7] + mode[10] - mode[11] + mode[13] - mode[14] + mode[16] + 2.f*mode[18]);
    //vd11
    buffer[3*nhyz + y + (para.dim_y*z)] = 1.f/36.f * (mode[0] + mode[1] + mode[3] + mode[4] + mode[5] - mode[6] + mode[8] + mode[10] + mode[12] - mode[13] + mode[15] + mode[16] + mode[17] - mode[18]);
    //vd13
    buffer[4*nhyz + y + (para.dim_y*z)] = 1.f/36.f * (mode[0] + mode[1] - mode[3] + mode[4] + mode[5] - mode[6] - mode[8] + mode[10] - mode[12] - mode[13] - mode[15] + mode[16] + mode[17] - mode[18]);
    //non buffered n's
  }
  //left buffered n's
  if(x == 0){
    //vd2
    buffer[0*nhyz + y + (para.dim_y*z) + 5*nhyz] = 1.f/18.f * (mode[0] - mode[1] + mode[5] + mode[6] - mode[17] - mode[18] + 2.f*(mode[10] - mode[16]));
    //vd8
    buffer[1*nhyz + y + (para.dim_y*z) + 5*nhyz] = 1.f/36.f * (mode[0] - mode[1] - mode[2] + mode[4] + 2.f*mode[6] + mode[7] - mode[10] - mode[11] - mode[13] - mode[14] + mode[16] + 2.f*mode[18]);
    //vd10
    buffer[2*nhyz + y + (para.dim_y*z) + 5*nhyz] = 1.f/36.f * (mode[0] - mode[1] + mode[2] + mode[4] + 2.f*mode[6] - mode[7] - mode[10] + mode[11] - mode[13] + mode[14] + mode[16] + 2.f*mode[18]);
    //vd12
    buffer[3*nhyz + y + (para.dim_y*z) + 5*nhyz] = 1.f/36.f * (mode[0] - mode[1] - mode[3] + mode[4] + mode[5] - mode[6] + mode[8] - mode[10] - mode[12] + mode[13] - mode[15] + mode[16] + mode[17] - mode[18]);
    //vd14
    buffer[4*nhyz + y + (para.dim_y*z) + 5*nhyz] = 1.f/36.f * (mode[0] - mode[1] + mode[3] + mode[4] + mode[5] - mode[6] - mode[8] - mode[10] + mode[12] + mode[13] + mode[15] + mode[16] + mode[17] - mode[18]);
  }
  //back buffered n's
  if(y == (para.dim_y-1)){
    //vd3
    buffer[0*nhxz + x + (para.dim_x*z) + 2*5*nhyz] = 1.f/18.f * (mode[0] + mode[2] - mode[5] + mode[6] + mode[17] - mode[18] - 2.f*(mode[11] + mode[16]));
    //vd7
    buffer[1*nhxz + x + (para.dim_x*z) + 2*5*nhyz] = 1.f/36.f * (mode[0] + mode[1] + mode[2] + mode[4] + 2.f*mode[6] + mode[7] + mode[10] + mode[11] + mode[13] + mode[14] + mode[16] + 2.f*mode[18]);
    //vd10
    buffer[2*nhxz + x + (para.dim_x*z) + 2*5*nhyz] = 1.f/36.f * (mode[0] - mode[1] + mode[2] + mode[4] + 2.f*mode[6] - mode[7] - mode[10] + mode[11] - mode[13] + mode[14] + mode[16] + 2.f*mode[18]);
    //vd15
    buffer[3*nhxz + x + (para.dim_x*z) + 2*5*nhyz] = 1.f/36.f * (mode[0] + mode[2] + mode[3] + mode[4] - mode[5] - mode[6] + mode[9] + mode[11] + mode[12] - mode[14] - mode[15] + mode[16] - mode[17] - mode[18]);
    //vd17
    buffer[4*nhxz + x + (para.dim_x*z) + 2*5*nhyz] = 1.f/36.f * (mode[0] + mode[2] - mode[3] + mode[4] - mode[5] - mode[6] - mode[9] + mode[11] - mode[12] - mode[14] + mode[15] + mode[16] - mode[17] - mode[18]);
  }
  //front buffered n's
  if(y == 0){
    //vd4
    buffer[0*nhxz + x + (para.dim_x*z) + 5*(2*nhyz + nhxz)] = 1.f/18.f * (mode[0] - mode[2] - mode[5] + mode[6] + mode[17] - mode[18] + 2.f*(mode[11] - mode[16]));
    //vd8
    buffer[1*nhxz + x + (para.dim_x*z) + 5*(2*nhyz + nhxz)] = 1.f/36.f * (mode[0] - mode[1] - mode[2] + mode[4] + 2.f*mode[6] + mode[7] - mode[10] - mode[11] - mode[13] - mode[14] + mode[16] + 2.f*mode[18]);
    //vd9
    buffer[2*nhxz + x + (para.dim_x*z) + 5*(2*nhyz + nhxz)] = 1.f/36.f * (mode[0] + mode[1] - mode[2] + mode[4] + 2.f*mode[6] - mode[7] + mode[10] - mode[11] + mode[13] - mode[14] + mode[16] + 2.f*mode[18]);
    //vd16
    buffer[3*nhxz + x + (para.dim_x*z) + 5*(2*nhyz + nhxz)] = 1.f/36.f * (mode[0] - mode[2] - mode[3] + mode[4] - mode[5] - mode[6] + mode[9] - mode[11] - mode[12] + mode[14] + mode[15] + mode[16] - mode[17] - mode[18]);
    //vd18
    buffer[4*nhxz + x + (para.dim_x*z) + 5*(2*nhyz + nhxz)] = 1.f/36.f * (mode[0] - mode[2] + mode[3] + mode[4] - mode[5] - mode[6] - mode[9] - mode[11] + mode[12] + mode[14] - mode[15] + mode[16] - mode[17] - mode[18]);
  }
  //up buffered n's
  if(z == (para.dim_z-1)){
    //vd5
    buffer[0*nhxy + x + (para.dim_x*y) + 5*2*(nhyz + nhxz)] = 1.f/18.f * (mode[0] + mode[3] - 2.f*(mode[6] + mode[12] + mode[16] - mode[18]));
    //vd11
    buffer[1*nhxy + x + (para.dim_x*y) + 5*2*(nhyz + nhxz)] = 1.f/36.f * (mode[0] + mode[1] + mode[3] + mode[4] + mode[5] - mode[6] + mode[8] + mode[10] + mode[12] - mode[13] + mode[15] + mode[16] + mode[17] - mode[18]);
    //vd14
    buffer[2*nhxy + x + (para.dim_x*y) + 5*2*(nhyz + nhxz)] = 1.f/36.f * (mode[0] - mode[1] + mode[3] + mode[4] + mode[5] - mode[6] - mode[8] - mode[10] + mode[12] + mode[13] + mode[15] + mode[16] + mode[17] - mode[18]);
    //vd15
    buffer[3*nhxy + x + (para.dim_x*y) + 5*2*(nhyz + nhxz)] = 1.f/36.f * (mode[0] + mode[2] + mode[3] + mode[4] - mode[5] - mode[6] + mode[9] + mode[11] + mode[12] - mode[14] - mode[15] + mode[16] - mode[17] - mode[18]);
    //vd18
    buffer[4*nhxy + x + (para.dim_x*y) + 5*2*(nhyz + nhxz)] = 1.f/36.f * (mode[0] - mode[2] + mode[3] + mode[4] - mode[5] - mode[6] - mode[9] - mode[11] + mode[12] + mode[14] - mode[15] + mode[16] - mode[17] - mode[18]);
  }
  //down buffered n's
  if(z == 0){
    //vd6
    buffer[0*nhxy + x + (para.dim_x*y) + 5*(2*(nhyz + nhxz) + nhxy)] = 1.f/18.f * (mode[0] - mode[3] - 2.f*(mode[6] - mode[12] + mode[16] - mode[18]));
    //vd12
    buffer[1*nhxy + x + (para.dim_x*y) + 5*(2*(nhyz + nhxz) + nhxy)] = 1.f/36.f * (mode[0] - mode[1] - mode[3] + mode[4] + mode[5] - mode[6] + mode[8] - mode[10] - mode[12] + mode[13] - mode[15] + mode[16] + mode[17] - mode[18]);
    //vd13
    buffer[2*nhxy + x + (para.dim_x*y) + 5*(2*(nhyz + nhxz) + nhxy)] = 1.f/36.f * (mode[0] + mode[1] - mode[3] + mode[4] + mode[5] - mode[6] - mode[8] + mode[10] - mode[12] - mode[13] - mode[15] + mode[16] + mode[17] - mode[18]);
    //vd16
    buffer[3*nhxy + x + (para.dim_x*y) + 5*(2*(nhyz + nhxz) + nhxy)] = 1.f/36.f * (mode[0] - mode[2] - mode[3] + mode[4] - mode[5] - mode[6] + mode[9] - mode[11] - mode[12] + mode[14] + mode[15] + mode[16] - mode[17] - mode[18]);
    //vd17
    buffer[4*nhxy + x + (para.dim_x*y) + 5*(2*(nhyz + nhxz) + nhxy)] = 1.f/36.f * (mode[0] + mode[2] - mode[3] + mode[4] - mode[5] - mode[6] - mode[9] + mode[11] - mode[12] - mode[14] + mode[15] + mode[16] - mode[17] - mode[18]);
  }
}

/*-------------------------------------------------------*/
/**write the received vds from buffer into their correct position in the nodes array 
 * @param index		node index / thread index (Input)
 * @param *n_b		Pointer to local node residing in array b (Output)
 * @param *buffer Pointer to buffer (Input)
 */
__device__ void write_n_from_buffer(LB_nodes_gpu n_b, float *buffer, unsigned int index){

  unsigned int xyz[3];
  index_to_xyz(index, xyz);
  unsigned int x = xyz[0];
  unsigned int y = xyz[1];
  unsigned int z = xyz[2];
  unsigned nhyz = para.number_of_halo_nodes[0];
  unsigned nhxz = para.number_of_halo_nodes[1];
  unsigned nhxy = para.number_of_halo_nodes[2];
  //right
  if(x == 0){
    //vd1
    n_b.vd[1*para.number_of_nodes + (x+1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*z] = buffer[0*nhyz + y + (para.dim_y*z)];
    //vd7
    n_b.vd[7*para.number_of_nodes + (x+1)%para.dim_x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*z] = buffer[1*nhyz + y + (para.dim_y*z)];
    //vd9
    n_b.vd[9*para.number_of_nodes + (x+1)%para.dim_x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*z] = buffer[2*nhyz + y + (para.dim_y*z)];
    //vd11
    n_b.vd[11*para.number_of_nodes + (x+1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*((z+1)%para.dim_z)] = buffer[3*nhyz + y + (para.dim_y*z)];
    //vd13
    n_b.vd[13*para.number_of_nodes + (x+1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z)] = buffer[4*nhyz + y + (para.dim_y*z)];
  }
  //left
  if(x == (para.dim_x-1)){
    //vd2
    n_b.vd[2*para.number_of_nodes + (para.dim_x+x-1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*z] = buffer[0*nhyz + y + (para.dim_y*z) + 5*nhyz];
    //vd8
    n_b.vd[8*para.number_of_nodes + (para.dim_x+x-1)%para.dim_x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*z] = buffer[1*nhyz + y + (para.dim_y*z) + 5*nhyz];
    //vd10
    n_b.vd[10*para.number_of_nodes + (para.dim_x+x-1)%para.dim_x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*z] = buffer[2*nhyz + y + (para.dim_y*z) + 5*nhyz];
    //vd12
    n_b.vd[12*para.number_of_nodes + (para.dim_x+x-1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z)] = buffer[3*nhyz + y + (para.dim_y*z) + 5*nhyz];
    //vd14
    n_b.vd[14*para.number_of_nodes + (para.dim_x+x-1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*((z+1)%para.dim_z)] = buffer[4*nhyz + y + (para.dim_y*z) + 5*nhyz];
  }
  //back
  if(y == 0){
    //vd3
    n_b.vd[3*para.number_of_nodes + x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*z] = buffer[0*nhxz + x + (para.dim_x*z) + 5*2*nhyz];
    //vd7
    n_b.vd[7*para.number_of_nodes + (x+1)%para.dim_x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*z] = buffer[1*nhxz + x + (para.dim_x*z) + 5*2*nhyz];
    //vd10
    n_b.vd[10*para.number_of_nodes + (para.dim_x+x-1)%para.dim_x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*z] = buffer[2*nhxz + x + (para.dim_x*z) + 5*2*nhyz];
    //vd15
    n_b.vd[15*para.number_of_nodes + x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*((z+1)%para.dim_z)] = buffer[3*nhxz + x + (para.dim_x*z) + 5*2*nhyz];
    //vd17
    n_b.vd[17*para.number_of_nodes + x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z)] = buffer[4*nhxz + x + (para.dim_x*z) + 5*2*nhyz];
  }
  //front
  if(y == (para.dim_y-1)){
    //vd4
    n_b.vd[4*para.number_of_nodes + x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*z] = buffer[0*nhxz + x + (para.dim_x*z) + 5*(2*nhyz + nhxz)];
    //vd8
    n_b.vd[8*para.number_of_nodes + (para.dim_x+x-1)%para.dim_x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*z] = buffer[1*nhxz + x + (para.dim_x*z) + 5*(2*nhyz + nhxz)];
    //vd9
    n_b.vd[9*para.number_of_nodes + (x+1)%para.dim_x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*z] = buffer[2*nhxz + x + (para.dim_x*z) + 5*(2*nhyz + nhxz)];
    //vd16
    n_b.vd[16*para.number_of_nodes + x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z)] = buffer[3*nhxz + x + (para.dim_x*z) + 5*(2*nhyz + nhxz)];
    //vd18
    n_b.vd[18*para.number_of_nodes + x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*((z+1)%para.dim_z)] = buffer[4*nhxz + x + (para.dim_x*z) + 5*(2*nhyz + nhxz)];
  }
  //up
  if(z == 0){
    //vd5
    n_b.vd[5*para.number_of_nodes + x + para.dim_x*y + para.dim_x*para.dim_y*((z+1)%para.dim_z)] = buffer[0*nhxy + x + (para.dim_x*y) + 5*2*(nhyz + nhxz)];
    //vd11
    n_b.vd[11*para.number_of_nodes + (x+1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*((z+1)%para.dim_z)] = buffer[1*nhxy + x + (para.dim_x*y) + 5*2*(nhyz + nhxz)];
    //vd14
    n_b.vd[14*para.number_of_nodes + (para.dim_x+x-1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*((z+1)%para.dim_z)] = buffer[2*nhxy + x + (para.dim_x*y) + 5*2*(nhyz + nhxz)];
    //vd15
    n_b.vd[15*para.number_of_nodes + x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*((z+1)%para.dim_z)] = buffer[3*nhxy + x + (para.dim_x*y) + 5*2*(nhyz + nhxz)];
    //vd18
    n_b.vd[18*para.number_of_nodes + x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*((z+1)%para.dim_z)] = buffer[4*nhxy + x + (para.dim_x*y) + 5*2*(nhyz + nhxz)];
  }
  //down
  if(z == (para.dim_z-1)){
    //vd6
    n_b.vd[6*para.number_of_nodes + x + para.dim_x*y + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z)] = buffer[0*nhxy + x + (para.dim_x*y) + 5*(2*(nhyz + nhxz) + nhxy)];
    //vd12
    n_b.vd[12*para.number_of_nodes + (para.dim_x+x-1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z)] = buffer[1*nhxy + x + (para.dim_x*y) + 5*(2*(nhyz + nhxz) + nhxy)];
    //vd13
    n_b.vd[13*para.number_of_nodes + (x+1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z)] = buffer[2*nhxy + x + (para.dim_x*y) + 5*(2*(nhyz + nhxz) + nhxy)];
    //vd16
    n_b.vd[16*para.number_of_nodes + x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z)] = buffer[3*nhxy + x + (para.dim_x*y) + 5*(2*(nhyz + nhxz) + nhxy)];
    //vd17
    n_b.vd[17*para.number_of_nodes + x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z)] = buffer[4*nhxy + x + (para.dim_x*y) + 5*(2*(nhyz + nhxz) + nhxy)];
  }
}
/** Bounce back boundary conditions.
 * The populations that have propagated into a boundary node
 * are bounced back to the node they came from. This results
 * in no slip boundary conditions.
 *
 * [cf. Ladd and Verberg, J. Stat. Phys. 104(5/6):1191-1251, 2001]
 * @param index			node index / thread index (Input)
 * @param n_b			Pointer to local node residing in array b (Input)
 * @param n_a			Pointer to local node residing in array a (Output) (temp stored in buffer a)
*/
__device__ void bounce_back_read(LB_nodes_gpu n_b, LB_nodes_gpu n_a, unsigned int index, \
    float* lb_boundary_velocity, float* lb_boundary_force){
    
  unsigned int xyz[3];
  int c[3];
  float v[3];
  float shift, weight, pop_to_bounce_back;
  float boundary_force[3] = {0,0,0};
  size_t to_index, to_index_x, to_index_y, to_index_z;
  int population, inverse;
  int boundary_index;


  boundary_index=n_b.boundary[index];
  if(boundary_index != 0){
    
    v[0]=lb_boundary_velocity[3*(boundary_index-1)+0];
    v[1]=lb_boundary_velocity[3*(boundary_index-1)+1];
    v[2]=lb_boundary_velocity[3*(boundary_index-1)+2];
    index_to_xyz(index, xyz);

    unsigned int x = xyz[0];
    unsigned int y = xyz[1];
    unsigned int z = xyz[2];

/* CPU analog of shift:
   lbpar.agrid*lbpar.agrid*lbpar.agrid*lbpar.rho*2*lbmodel.c[i][l]*lb_boundaries[lbfields[k].boundary-1].velocity[l] */
  
    /** store vd temporary in second lattice to avoid race conditions */
#define BOUNCEBACK  \
  shift = para.agrid*para.agrid*para.agrid*para.agrid*para.rho*2.*3.*weight*para.tau*(v[0]*c[0] + v[1]*c[1] + v[2]*c[2]); \
  pop_to_bounce_back = n_b.vd[population*para.number_of_nodes + index ]; \
  to_index_x = (x+c[0]+para.dim_x)%para.dim_x; \
  to_index_y = (y+c[1]+para.dim_y)%para.dim_y; \
  to_index_z = (z+c[2]+para.dim_z)%para.dim_z; \
  to_index = to_index_x + para.dim_x*to_index_y + para.dim_x*para.dim_y*to_index_z; \
  if (n_b.boundary[to_index] == 0) \
  { \
    boundary_force[0] += (2*pop_to_bounce_back+shift)*c[0]/para.tau/para.tau/para.agrid; \
    boundary_force[1] += (2*pop_to_bounce_back+shift)*c[1]/para.tau/para.tau/para.agrid; \
    boundary_force[2] += (2*pop_to_bounce_back+shift)*c[2]/para.tau/para.tau/para.agrid; \
    n_b.vd[inverse*para.number_of_nodes + to_index ] = pop_to_bounce_back + shift; \
  }

// ***** SHOULDN'T THERE BE AN ELSE STATMENT IN "BOUNCEBACK"?
// ***** THERE IS AN ODD FACTOR OF 2 THAT YOU INCUR IN THE FORCES FOR THE "lb_stokes_sphere_gpu.tcl" TEST CASE

/*        printf("boundary %d population %d creates force  %f %f %f\n", boundary_index-1, population, 1000*pop_to_bounce_back*c[0],  1000*pop_to_bounce_back*c[1],  1000*pop_to_bounce_back*c[2]);\
      } */

// ***** WHAT DO THE "to_index" STATMENTS? THEY GET WRITTEN OVER BY "BOUNCEBACK" ANYWAY? I THINK THESE CAN BE REMOVED, COMMENTING THEM OUT DID NOT SEEM TO EFFECT ANYTHING.

    // the resting population does nothing.
    c[0]=1;c[1]=0;c[2]=0; weight=1./18.; population=2; inverse=1; 
//    to_index=(x+1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*z;  
    BOUNCEBACK
    
    c[0]=-1;c[1]=0;c[2]=0; weight=1./18.; population=1; inverse=2; 
//    to_index=(para.dim_x+x-1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*z;  
    BOUNCEBACK
    
    c[0]=0;c[1]=1;c[2]=0;  weight=1./18.; population=4; inverse=3; 
 //   to_index= x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*z;  
    BOUNCEBACK

    c[0]=0;c[1]=-1;c[2]=0; weight=1./18.; population=3; inverse=4; 
//    to_index=x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*z;  
    BOUNCEBACK
    
    c[0]=0;c[1]=0;c[2]=1; weight=1./18.; population=6; inverse=5; 
//    to_index=x + para.dim_x*y + para.dim_x*para.dim_y*((z+1)%para.dim_z);  
    BOUNCEBACK

    c[0]=0;c[1]=0;c[2]=-1; weight=1./18.; population=5; inverse=6; 
 //   to_index=x + para.dim_x*y + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z);  
    BOUNCEBACK 
    
    c[0]=1;c[1]=1;c[2]=0; weight=1./36.; population=8; inverse=7; 
//    to_index=+(x+1)%para.dim_x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*z;  
    BOUNCEBACK
    
    c[0]=-1;c[1]=-1;c[2]=0; weight=1./36.; population=7; inverse=8; 
//    to_index= (para.dim_x+x-1)%para.dim_x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*z;  
    BOUNCEBACK
    
    c[0]=1;c[1]=-1;c[2]=0; weight=1./36.; population=10; inverse=9; 
//    to_index= (x+1)%para.dim_x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*z;  
    BOUNCEBACK

    c[0]=-1;c[1]=+1;c[2]=0; weight=1./36.; population=9; inverse=10; 
//    to_index= + (para.dim_x+x-1)%para.dim_x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*z;  
    BOUNCEBACK
    
    c[0]=1;c[1]=0;c[2]=1; weight=1./36.; population=12; inverse=11; 
//    to_index= + (x+1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*((z+1)%para.dim_z);  
    BOUNCEBACK
    
    c[0]=-1;c[1]=0;c[2]=-1; weight=1./36.; population=11; inverse=12; 
 //   to_index= + (para.dim_x+x-1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z);  
    BOUNCEBACK

    c[0]=1;c[1]=0;c[2]=-1; weight=1./36.; population=14; inverse=13; 
 //   to_index= + (x+1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z);  
    BOUNCEBACK
    
    c[0]=-1;c[1]=0;c[2]=1; weight=1./36.; population=13; inverse=14; 
  //  to_index=(para.dim_x+x-1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*((z+1)%para.dim_z);  
    BOUNCEBACK

    c[0]=0;c[1]=1;c[2]=1; weight=1./36.; population=16; inverse=15; 
 //   to_index= + x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*((z+1)%para.dim_z);  
    BOUNCEBACK
    
    c[0]=0;c[1]=-1;c[2]=-1; weight=1./36.; population=15; inverse=16; 
//    to_index=+ x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z);  
    BOUNCEBACK
    
    c[0]=0;c[1]=1;c[2]=-1; weight=1./36.; population=18; inverse=17; 
 //   to_index=+ x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z); 
    BOUNCEBACK
    
    c[0]=0;c[1]=-1;c[2]=1; weight=1./36.; population=17; inverse=18; 
//    to_index= + x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*((z+1)%para.dim_z);  
    BOUNCEBACK  
    
  atomicadd(&lb_boundary_force[3*(n_b.boundary[index]-1)+0], boundary_force[0]);
  atomicadd(&lb_boundary_force[3*(n_b.boundary[index]-1)+1], boundary_force[1]);
  atomicadd(&lb_boundary_force[3*(n_b.boundary[index]-1)+2], boundary_force[2]);
  }
}
/**bounce back read kernel needed to avoid raceconditions
 * @param index			node index / thread index (Input)
 * @param n_b			Pointer to local node residing in array b (Input)
 * @param n_a			Pointer to local node residing in array a (Output) (temp stored in buffer a)
*/
__device__ void bounce_back_write(LB_nodes_gpu n_b, LB_nodes_gpu n_a, unsigned int index){

  unsigned int xyz[3];

  if(n_b.boundary[index] != 0){
    index_to_xyz(index, xyz);
    unsigned int x = xyz[0];
    unsigned int y = xyz[1];
    unsigned int z = xyz[2];

    /** stream vd from boundary node back to origin node */
    n_b.vd[1*para.number_of_nodes + (x+1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*z] = n_a.vd[1*para.number_of_nodes + (x+1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*z];
    n_b.vd[2*para.number_of_nodes + (para.dim_x+x-1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*z] = n_a.vd[2*para.number_of_nodes + (para.dim_x+x-1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*z];
    n_b.vd[3*para.number_of_nodes + x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*z] = n_a.vd[3*para.number_of_nodes + x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*z];
    n_b.vd[4*para.number_of_nodes + x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*z] = n_a.vd[4*para.number_of_nodes + x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*z];
    n_b.vd[5*para.number_of_nodes + x + para.dim_x*y + para.dim_x*para.dim_y*((z+1)%para.dim_z)] = n_a.vd[5*para.number_of_nodes + x + para.dim_x*y + para.dim_x*para.dim_y*((z+1)%para.dim_z)];
    n_b.vd[6*para.number_of_nodes + x + para.dim_x*y + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z)] = n_a.vd[6*para.number_of_nodes + x + para.dim_x*y + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z)];
    n_b.vd[7*para.number_of_nodes + (x+1)%para.dim_x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*z] = n_a.vd[7*para.number_of_nodes + (x+1)%para.dim_x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*z];
    n_b.vd[8*para.number_of_nodes + (para.dim_x+x-1)%para.dim_x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*z] = n_a.vd[8*para.number_of_nodes + (para.dim_x+x-1)%para.dim_x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*z];
    n_b.vd[9*para.number_of_nodes + (x+1)%para.dim_x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*z] = n_a.vd[9*para.number_of_nodes + (x+1)%para.dim_x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*z];
    n_b.vd[10*para.number_of_nodes + (para.dim_x+x-1)%para.dim_x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*z] = n_a.vd[10*para.number_of_nodes + (para.dim_x+x-1)%para.dim_x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*z];
    n_b.vd[11*para.number_of_nodes + (x+1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*((z+1)%para.dim_z)] = n_a.vd[11*para.number_of_nodes + (x+1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*((z+1)%para.dim_z)];
    n_b.vd[12*para.number_of_nodes + (para.dim_x+x-1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z)] = n_a.vd[12*para.number_of_nodes + (para.dim_x+x-1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z)];
    n_b.vd[13*para.number_of_nodes + (x+1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z)] = n_a.vd[13*para.number_of_nodes + (x+1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z)];
    n_b.vd[14*para.number_of_nodes + (para.dim_x+x-1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*((z+1)%para.dim_z)] = n_a.vd[14*para.number_of_nodes + (para.dim_x+x-1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*((z+1)%para.dim_z)];
    n_b.vd[15*para.number_of_nodes + x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*((z+1)%para.dim_z)] = n_a.vd[15*para.number_of_nodes + x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*((z+1)%para.dim_z)];
    n_b.vd[16*para.number_of_nodes + x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z)] = n_a.vd[16*para.number_of_nodes + x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z)];
    n_b.vd[17*para.number_of_nodes + x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z)] = n_a.vd[17*para.number_of_nodes + x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z)];
    n_b.vd[18*para.number_of_nodes + x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*((z+1)%para.dim_z)] = n_a.vd[18*para.number_of_nodes + x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*((z+1)%para.dim_z)];
  }
}
/** add of (external) forces within the modespace, needed for particle-interaction
 * @param index		node index / thread index (Input)
 * @param mode		Pointer to the local register values mode (Input/Output)
 * @param node_f	Pointer to local node force (Input)
*/
__device__ void apply_forces(unsigned int index, float *mode, LB_node_force_gpu node_f) {

  float Rho, u[3], C[6];
  Rho = mode[0] + para.rho*para.agrid*para.agrid*para.agrid;

  /** hydrodynamic momentum density is redefined when forces present */
  u[0] = (mode[1] + 0.5f*node_f.force[0*para.number_of_nodes + index])/Rho;
  u[1] = (mode[2] + 0.5f*node_f.force[1*para.number_of_nodes + index])/Rho;
  u[2] = (mode[3] + 0.5f*node_f.force[2*para.number_of_nodes + index])/Rho;

  C[0] = (1.f + para.gamma_bulk)*u[0]*node_f.force[0*para.number_of_nodes + index] + 1.f/3.f*(para.gamma_bulk-para.gamma_shear)*(u[0]*node_f.force[0*para.number_of_nodes + index] + u[1]*node_f.force[1*para.number_of_nodes + index] + u[2]*node_f.force[2*para.number_of_nodes + index]);
  C[2] = (1.f + para.gamma_bulk)*u[1]*node_f.force[1*para.number_of_nodes + index] + 1.f/3.f*(para.gamma_bulk-para.gamma_shear)*(u[0]*node_f.force[0*para.number_of_nodes + index] + u[1]*node_f.force[1*para.number_of_nodes + index] + u[2]*node_f.force[2*para.number_of_nodes + index]);
  C[5] = (1.f + para.gamma_bulk)*u[2]*node_f.force[2*para.number_of_nodes + index] + 1.f/3.f*(para.gamma_bulk-para.gamma_shear)*(u[0]*node_f.force[0*para.number_of_nodes + index] + u[1]*node_f.force[1*para.number_of_nodes + index] + u[2]*node_f.force[2*para.number_of_nodes + index]);
  C[1] = 1.f/2.f*(1.f+para.gamma_shear)*(u[0]*node_f.force[1*para.number_of_nodes + index]+u[1]*node_f.force[0*para.number_of_nodes + index]);
  C[3] = 1.f/2.f*(1.f+para.gamma_shear)*(u[0]*node_f.force[2*para.number_of_nodes + index]+u[2]*node_f.force[0*para.number_of_nodes + index]);
  C[4] = 1.f/2.f*(1.f+para.gamma_shear)*(u[1]*node_f.force[2*para.number_of_nodes + index]+u[2]*node_f.force[1*para.number_of_nodes + index]);

  /** update momentum modes */
  mode[1] += node_f.force[0*para.number_of_nodes + index];
  mode[2] += node_f.force[1*para.number_of_nodes + index];
  mode[3] += node_f.force[2*para.number_of_nodes + index];

  /** update stress modes */
  mode[4] += C[0] + C[2] + C[5];
  mode[5] += C[0] - C[2];
  mode[6] += C[0] + C[2] - 2.f*C[5];
  mode[7] += C[1];
  mode[8] += C[3];
  mode[9] += C[4];

#ifdef EXTERNAL_FORCES
  if(para.external_force){
    node_f.force[0*para.number_of_nodes + index] = para.ext_force[0]*powf(para.agrid,4)*para.tau*para.tau;
    node_f.force[1*para.number_of_nodes + index] = para.ext_force[1]*powf(para.agrid,4)*para.tau*para.tau;
    node_f.force[2*para.number_of_nodes + index] = para.ext_force[2]*powf(para.agrid,4)*para.tau*para.tau;
  }
  else{
  node_f.force[0*para.number_of_nodes + index] = 0.f;
  node_f.force[1*para.number_of_nodes + index] = 0.f;
  node_f.force[2*para.number_of_nodes + index] = 0.f;
  }
#else
  /** reset force */
  node_f.force[0*para.number_of_nodes + index] = 0.f;
  node_f.force[1*para.number_of_nodes + index] = 0.f;
  node_f.force[2*para.number_of_nodes + index] = 0.f;
#endif
}

/**function used to calc physical values of every node
 * @param index		node index / thread index (Input)
 * @param mode		Pointer to the local register values mode (Input)
 * @param n_a		Pointer to local node residing in array a for boundary flag(Input)
 * @param *d_v		Pointer to local device values (Input/Output)
 * @param singlenode	Flag, if there is only one node
 * This function is a clone of lb_calc_local_fields and
 * additionally performs unit conversions.
*/
__device__ void calc_values(LB_nodes_gpu n_a, float *mode, LB_values_gpu *d_v, unsigned int index, unsigned int singlenode, LB_node_force_gpu node_f){

  float rho = mode[0] + para.rho*para.agrid*para.agrid*para.agrid;

  float *v, *pi;
  if(singlenode == 1){ 
    v=&(d_v[0].v[0]);
    pi=&(d_v[0].pi[0]);
    d_v[0].rho = rho;
  } else {
    v=&(d_v[index].v[0]);
    pi=&(d_v[index].pi[0]);
    d_v[index].rho = rho;
  }
  float j[3]; float pi_eq[6];
  
  j[0] = mode[1];
  j[1] = mode[2];
  j[2] = mode[3];

  j[0] += 0.5*node_f.force[0*para.number_of_nodes + index];
  j[1] += 0.5*node_f.force[1*para.number_of_nodes + index];
  j[2] += 0.5*node_f.force[2*para.number_of_nodes + index];

  v[0]=j[0]/rho;
  v[1]=j[1]/rho;
  v[2]=j[2]/rho;

  /* equilibrium part of the stress modes */
  pi_eq[0] = (j[0]*j[0]+j[1]*j[1]+j[2]*j[2])/ rho;
  pi_eq[1] = ((j[0]*j[0])-(j[1]*j[1]))/ rho;
  pi_eq[2] = (j[0]*j[0]+j[1]*j[1]+j[2]*j[2] - 3.0*j[2]*j[2])/ rho;
  pi_eq[3] = j[0]*j[1]/ rho;
  pi_eq[4] = j[0]*j[2]/ rho;
  pi_eq[5] = j[1]*j[2]/ rho;

  /* Now we must predict the outcome of the next collision */
  /* We immediately average pre- and post-collision. */
  mode[4] = pi_eq[0] + (0.5+0.5*para.gamma_bulk )*(mode[4] - pi_eq[0]);
  mode[5] = pi_eq[1] + (0.5+0.5*para.gamma_shear)*(mode[5] - pi_eq[1]);
  mode[6] = pi_eq[2] + (0.5+0.5*para.gamma_shear)*(mode[6] - pi_eq[2]);
  mode[7] = pi_eq[3] + (0.5+0.5*para.gamma_shear)*(mode[7] - pi_eq[3]);
  mode[8] = pi_eq[4] + (0.5+0.5*para.gamma_shear)*(mode[8] - pi_eq[4]);
  mode[9] = pi_eq[5] + (0.5+0.5*para.gamma_shear)*(mode[9] - pi_eq[5]);

  /* Now we have to transform to the "usual" stress tensor components */
  /* We use eq. 116ff in Duenweg Ladd for that. */
  pi[0]=(mode[0]+mode[4]+mode[5])/3.;
  pi[2]=(2*mode[0]+2*mode[4]-mode[5]+3*mode[6])/6.;
  pi[5]=(2*mode[0]+2*mode[4]-mode[5]+3*mode[6])/6.;
  pi[1]=mode[7];
  pi[3]=mode[8];
  pi[4]=mode[9];

  /* Finally some unit conversions */
  rho*=1./para.agrid/para.agrid/para.agrid;
  v[0]*=1./para.tau/para.agrid;
  v[1]*=1./para.tau/para.agrid;
  v[2]*=1./para.tau/para.agrid;

  for (int i =0; i<6; i++) {
    pi[i]*=1./para.tau/para.tau/para.agrid/para.agrid/para.agrid;
  }

}
/**function used to calc physical values of every node
 * @param index		node index / thread index (Input)
 * @param mode		Pointer to the local register values mode (Input)
 * @param n_a		Pointer to local node residing in array a for boundary flag(Input)
 * @param *d_v		Pointer to local device values (Input/Output)
 * @param singlenode	Flag, if there is only one node
 * This function is a clone of lb_calc_local_fields and
 * additionally performs unit conversions.
*/
__device__ void calc_values_wo_halo(LB_nodes_gpu n_a, float *mode, LB_values_gpu *d_v, unsigned int index, unsigned int singlenode, LB_node_force_gpu node_f){

    unsigned int iindex;
    unsigned int xyz[3];
    index_to_xyz(index, xyz);
    unsigned int x = xyz[0];
    unsigned int y = xyz[1];
    unsigned int z = xyz[2];
    iindex=index;
    if(x != 0 && x != (para.dim_x-1) && y != 0 && y != (para.dim_y-1) && z != 0 && z != (para.dim_z-1)){
     iindex = (x-1) + (y-1)*(para.dim_x-2) + (z-1)*(para.dim_x-2)*(para.dim_y-2);
     float rho = mode[0] + para.rho*para.agrid*para.agrid*para.agrid;
     float *v, *pi;
     if(singlenode == 1){ 
      v=&(d_v[0].v[0]);
      pi=&(d_v[0].pi[0]);
      d_v[0].rho = rho;
     } else {
       //TODO compute correct index
         v=&(d_v[iindex].v[0]);
         pi=&(d_v[iindex].pi[0]);
         d_v[iindex].rho = rho;
       }
     float j[3]; float pi_eq[6];
  
     j[0] = mode[1];
     j[1] = mode[2];
     j[2] = mode[3];

    j[0] += 0.5*node_f.force[0*para.number_of_nodes + index];
    j[1] += 0.5*node_f.force[1*para.number_of_nodes + index];
    j[2] += 0.5*node_f.force[2*para.number_of_nodes + index];

    v[0]=j[0]/rho;
    v[1]=j[1]/rho;
    v[2]=j[2]/rho;

    /* equilibrium part of the stress modes */
    pi_eq[0] = (j[0]*j[0]+j[1]*j[1]+j[2]*j[2])/ rho;
    pi_eq[1] = ((j[0]*j[0])-(j[1]*j[1]))/ rho;
    pi_eq[2] = (j[0]*j[0]+j[1]*j[1]+j[2]*j[2] - 3.0*j[2]*j[2])/ rho;
    pi_eq[3] = j[0]*j[1]/ rho;
    pi_eq[4] = j[0]*j[2]/ rho;
    pi_eq[5] = j[1]*j[2]/ rho;

    /* Now we must predict the outcome of the next collision */
    /* We immediately average pre- and post-collision. */
    mode[4] = pi_eq[0] + (0.5+0.5*para.gamma_bulk )*(mode[4] - pi_eq[0]);
    mode[5] = pi_eq[1] + (0.5+0.5*para.gamma_shear)*(mode[5] - pi_eq[1]);
    mode[6] = pi_eq[2] + (0.5+0.5*para.gamma_shear)*(mode[6] - pi_eq[2]);
    mode[7] = pi_eq[3] + (0.5+0.5*para.gamma_shear)*(mode[7] - pi_eq[3]);
    mode[8] = pi_eq[4] + (0.5+0.5*para.gamma_shear)*(mode[8] - pi_eq[4]);
    mode[9] = pi_eq[5] + (0.5+0.5*para.gamma_shear)*(mode[9] - pi_eq[5]);

    /* Now we have to transform to the "usual" stress tensor components */
    /* We use eq. 116ff in Duenweg Ladd for that. */
    pi[0]=(mode[0]+mode[4]+mode[5])/3.;
    pi[2]=(2*mode[0]+2*mode[4]-mode[5]+3*mode[6])/6.;
    pi[5]=(2*mode[0]+2*mode[4]-mode[5]+3*mode[6])/6.;
    pi[1]=mode[7];
    pi[3]=mode[8];
    pi[4]=mode[9];

    /* Finally some unit conversions */
    rho*=1./para.agrid/para.agrid/para.agrid;
    v[0]*=1./para.tau/para.agrid;
    v[1]*=1./para.tau/para.agrid;
    v[2]*=1./para.tau/para.agrid;
    for (int i =0; i<6; i++) {
      pi[i]*=1./para.tau/para.tau/para.agrid/para.agrid/para.agrid;
    }
  }

}
/** calculates only 4 conserved modes
 * @param node_index	node index around (8) particle (Input)
 * @param *mode			Pointer to the local register values mode (Output)
 * @param n_a			Pointer to local node residing in array a(Input)
*/
__device__ void calc_mode(float *mode, LB_nodes_gpu n_a, unsigned int node_index){

  /** mass mode */
  mode[0] = n_a.vd[0*para.number_of_nodes + node_index] + n_a.vd[1*para.number_of_nodes + node_index] + n_a.vd[2*para.number_of_nodes + node_index] 
          + n_a.vd[3*para.number_of_nodes + node_index] + n_a.vd[4*para.number_of_nodes + node_index] + n_a.vd[5*para.number_of_nodes + node_index]
          + n_a.vd[6*para.number_of_nodes + node_index] + n_a.vd[7*para.number_of_nodes + node_index] + n_a.vd[8*para.number_of_nodes + node_index]
          + n_a.vd[9*para.number_of_nodes + node_index] + n_a.vd[10*para.number_of_nodes + node_index] + n_a.vd[11*para.number_of_nodes + node_index] + n_a.vd[12*para.number_of_nodes + node_index]
          + n_a.vd[13*para.number_of_nodes + node_index] + n_a.vd[14*para.number_of_nodes + node_index] + n_a.vd[15*para.number_of_nodes + node_index] + n_a.vd[16*para.number_of_nodes + node_index]
          + n_a.vd[17*para.number_of_nodes + node_index] + n_a.vd[18*para.number_of_nodes + node_index];

  /** momentum modes */
  mode[1] = (n_a.vd[1*para.number_of_nodes + node_index] - n_a.vd[2*para.number_of_nodes + node_index]) + (n_a.vd[7*para.number_of_nodes + node_index] - n_a.vd[8*para.number_of_nodes + node_index])
          + (n_a.vd[9*para.number_of_nodes + node_index] - n_a.vd[10*para.number_of_nodes + node_index]) + (n_a.vd[11*para.number_of_nodes + node_index] - n_a.vd[12*para.number_of_nodes + node_index])
          + (n_a.vd[13*para.number_of_nodes + node_index] - n_a.vd[14*para.number_of_nodes + node_index]);
  mode[2] = (n_a.vd[3*para.number_of_nodes + node_index] - n_a.vd[4*para.number_of_nodes + node_index]) + (n_a.vd[7*para.number_of_nodes + node_index] - n_a.vd[8*para.number_of_nodes + node_index])
          - (n_a.vd[9*para.number_of_nodes + node_index] - n_a.vd[10*para.number_of_nodes + node_index]) + (n_a.vd[15*para.number_of_nodes + node_index] - n_a.vd[16*para.number_of_nodes + node_index])
          + (n_a.vd[17*para.number_of_nodes + node_index] - n_a.vd[18*para.number_of_nodes + node_index]);
  mode[3] = (n_a.vd[5*para.number_of_nodes + node_index] - n_a.vd[6*para.number_of_nodes + node_index]) + (n_a.vd[11*para.number_of_nodes + node_index] - n_a.vd[12*para.number_of_nodes + node_index])
          - (n_a.vd[13*para.number_of_nodes + node_index] - n_a.vd[14*para.number_of_nodes + node_index]) + (n_a.vd[15*para.number_of_nodes + node_index] - n_a.vd[16*para.number_of_nodes + node_index])
          - (n_a.vd[17*para.number_of_nodes + node_index] - n_a.vd[18*para.number_of_nodes + node_index]);
}
#ifndef SHANCHEN
/**calculate temperature of the fluid kernel
 * @param *cpu_jsquared			Pointer to result storage value (Output)
 * @param n_a				Pointer to local node residing in array a (Input)
*/
__global__ void temperature(LB_nodes_gpu n_a, float *cpu_jsquared) {
  float mode[4];
  float jsquared = 0.f;
  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x;

  if(index<para.number_of_nodes){
    calc_mode(mode, n_a, index);
    if(n_a.boundary[index]){
      jsquared = 0.f;
    }
    else{
      jsquared = mode[1]*mode[1]+mode[2]*mode[2]+mode[3]*mode[3];
    }
    atomicadd(cpu_jsquared, jsquared);
  }
}
#endif

/*********************************************************/
/** \name Coupling part */
/*********************************************************/
/**(Eq. (12) Ahlrichs and Duenweg, JCP 111(17):8225 (1999))
 * @param n_a			Pointer to local node residing in array a (Input)
 * @param *delta		Pointer for the weighting of particle position (Output)
 * @param *delta_j		Pointer for the weighting of particle momentum (Output)
 * @param *particle_data	Pointer to the particle position and velocity (Input)
 * @param *particle_force	Pointer to the particle force (Input)
 * @param part_index		particle id / thread id (Input)
 * @param *rn_part		Pointer to randomnumber array of the particle
 * @param node_index		node index around (8) particle (Output)
*/
#ifndef SHANCHEN
__device__ void calc_viscous_force(LB_nodes_gpu n_a, float *delta, LB_particle_gpu *particle_data, LB_particle_force_gpu *particle_force, unsigned int part_index, LB_randomnr_gpu *rn_part, float *delta_j, unsigned int *node_index){

  float mode[4];
  int my_left[3];
  float interpolated_u1, interpolated_u2, interpolated_u3;
  float Rho;
  interpolated_u1 = interpolated_u2 = interpolated_u3 = 0.f;

  float temp_delta[6];
  float temp_delta_half[6];

  /** see ahlrichs + duenweg page 8227 equ (10) and (11) */
  #pragma unroll
  for(int i=0; i<3; ++i){
    float scaledpos = (particle_data[part_index].p[i]-0.5f)/para.agrid;
    my_left[i] = (int)(floorf(scaledpos));
    //temp_delta[3+i] = scaledpos - my_left[i];
    temp_delta[3+i] = scaledpos - my_left[i];
    temp_delta[i] = 1.f - temp_delta[3+i];
    /**further value used for interpolation of fluid velocity at part pos near boundaries */
    temp_delta_half[3+i] = (scaledpos - my_left[i])*2.f;
    temp_delta_half[i] = 2.f - temp_delta_half[3+i];
  }

  delta[0] = temp_delta[0] * temp_delta[1] * temp_delta[2];
  delta[1] = temp_delta[3] * temp_delta[1] * temp_delta[2];
  delta[2] = temp_delta[0] * temp_delta[4] * temp_delta[2];
  delta[3] = temp_delta[3] * temp_delta[4] * temp_delta[2];
  delta[4] = temp_delta[0] * temp_delta[1] * temp_delta[5];
  delta[5] = temp_delta[3] * temp_delta[1] * temp_delta[5];
  delta[6] = temp_delta[0] * temp_delta[4] * temp_delta[5];
  delta[7] = temp_delta[3] * temp_delta[4] * temp_delta[5];

  // modulo for negative numbers is strange at best, shift to make sure we are positive
  int x = my_left[0] + para.dim_x;
  int y = my_left[1] + para.dim_y;
  int z = my_left[2] + para.dim_z;

  node_index[0] = x%para.dim_x     + para.dim_x*(y%para.dim_y)     + para.dim_x*para.dim_y*(z%para.dim_z);
  node_index[1] = (x+1)%para.dim_x + para.dim_x*(y%para.dim_y)     + para.dim_x*para.dim_y*(z%para.dim_z);
  node_index[2] = x%para.dim_x     + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*(z%para.dim_z);
  node_index[3] = (x+1)%para.dim_x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*(z%para.dim_z);
  node_index[4] = x%para.dim_x     + para.dim_x*(y%para.dim_y)     + para.dim_x*para.dim_y*((z+1)%para.dim_z);
  node_index[5] = (x+1)%para.dim_x + para.dim_x*(y%para.dim_y)     + para.dim_x*para.dim_y*((z+1)%para.dim_z);
  node_index[6] = x%para.dim_x     + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*((z+1)%para.dim_z);
  node_index[7] = (x+1)%para.dim_x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*((z+1)%para.dim_z);
  #pragma unroll
  for(int i=0; i<8; ++i){
    calc_mode(mode, n_a, node_index[i]);
    Rho = mode[0] + para.rho*para.agrid*para.agrid*para.agrid;
    interpolated_u1 += delta[i]*mode[1]/(Rho);
    interpolated_u2 += delta[i]*mode[2]/(Rho);
    interpolated_u3 += delta[i]*mode[3]/(Rho);
  }

  /** calculate viscous force
   * take care to rescale velocities with time_step and transform to MD units
   * (Eq. (9) Ahlrichs and Duenweg, JCP 111(17):8225 (1999)) */
#ifdef LB_ELECTROHYDRODYNAMICS
  particle_force[part_index].f[0] = - para.friction * (particle_data[part_index].v[0]/para.time_step - interpolated_u1*para.agrid/para.tau - particle_data[part_index].mu_E[0]);
  particle_force[part_index].f[1] = - para.friction * (particle_data[part_index].v[1]/para.time_step - interpolated_u2*para.agrid/para.tau - particle_data[part_index].mu_E[1]);
  particle_force[part_index].f[2] = - para.friction * (particle_data[part_index].v[2]/para.time_step - interpolated_u3*para.agrid/para.tau - particle_data[part_index].mu_E[2]);
#else
  particle_force[part_index].f[0] = - para.friction * (particle_data[part_index].v[0]/para.time_step - interpolated_u1*para.agrid/para.tau);
  particle_force[part_index].f[1] = - para.friction * (particle_data[part_index].v[1]/para.time_step - interpolated_u2*para.agrid/para.tau);
  particle_force[part_index].f[2] = - para.friction * (particle_data[part_index].v[2]/para.time_step - interpolated_u3*para.agrid/para.tau);
#endif
  /** add stochastic force of zero mean (Ahlrichs, Duenweg equ. 15)*/
#ifdef GAUSSRANDOM
  gaussian_random(rn_part);
  particle_force[part_index].f[0] += para.lb_coupl_pref2*rn_part->randomnr[0];
  particle_force[part_index].f[1] += para.lb_coupl_pref2*rn_part->randomnr[1];
  gaussian_random(rn_part);
  particle_force[part_index].f[2] += para.lb_coupl_pref2*rn_part->randomnr[0];
#else
  random_01(rn_part);
  particle_force[part_index].f[0] += para.lb_coupl_pref*(rn_part->randomnr[0]-0.5f);
  particle_force[part_index].f[1] += para.lb_coupl_pref*(rn_part->randomnr[1]-0.5f);
  random_01(rn_part);
  particle_force[part_index].f[2] += para.lb_coupl_pref*(rn_part->randomnr[0]-0.5f);
#endif
  /** delta_j for transform momentum transfer to lattice units which is done in calc_node_force
  (Eq. (12) Ahlrichs and Duenweg, JCP 111(17):8225 (1999)) */
  delta_j[0] = - particle_force[part_index].f[0]*para.time_step*para.tau/para.agrid;
  delta_j[1] = - particle_force[part_index].f[1]*para.time_step*para.tau/para.agrid;
  delta_j[2] = - particle_force[part_index].f[2]*para.time_step*para.tau/para.agrid;  	

}
#endif
/**calcutlation of the node force caused by the particles, with atomicadd due to avoiding race conditions
	(Eq. (14) Ahlrichs and Duenweg, JCP 111(17):8225 (1999))
 * @param *delta		Pointer for the weighting of particle position (Input)
 * @param *delta_j		Pointer for the weighting of particle momentum (Input)
 * @param node_index		node index around (8) particle (Input)
 * @param node_f    		Pointer to the node force (Output).
*/
__device__ void calc_node_force(float *delta, float *delta_j, unsigned int *node_index, LB_node_force_gpu node_f){

  atomicadd(&(node_f.force[0*para.number_of_nodes + node_index[0]]), (delta[0]*delta_j[0]));
  atomicadd(&(node_f.force[1*para.number_of_nodes + node_index[0]]), (delta[0]*delta_j[1]));
  atomicadd(&(node_f.force[2*para.number_of_nodes + node_index[0]]), (delta[0]*delta_j[2]));

  atomicadd(&(node_f.force[0*para.number_of_nodes + node_index[1]]), (delta[1]*delta_j[0]));
  atomicadd(&(node_f.force[1*para.number_of_nodes + node_index[1]]), (delta[1]*delta_j[1]));
  atomicadd(&(node_f.force[2*para.number_of_nodes + node_index[1]]), (delta[1]*delta_j[2]));

  atomicadd(&(node_f.force[0*para.number_of_nodes + node_index[2]]), (delta[2]*delta_j[0]));
  atomicadd(&(node_f.force[1*para.number_of_nodes + node_index[2]]), (delta[2]*delta_j[1]));
  atomicadd(&(node_f.force[2*para.number_of_nodes + node_index[2]]), (delta[2]*delta_j[2]));

  atomicadd(&(node_f.force[0*para.number_of_nodes + node_index[3]]), (delta[3]*delta_j[0]));
  atomicadd(&(node_f.force[1*para.number_of_nodes + node_index[3]]), (delta[3]*delta_j[1]));
  atomicadd(&(node_f.force[2*para.number_of_nodes + node_index[3]]), (delta[3]*delta_j[2]));

  atomicadd(&(node_f.force[0*para.number_of_nodes + node_index[4]]), (delta[4]*delta_j[0]));
  atomicadd(&(node_f.force[1*para.number_of_nodes + node_index[4]]), (delta[4]*delta_j[1]));
  atomicadd(&(node_f.force[2*para.number_of_nodes + node_index[4]]), (delta[4]*delta_j[2]));

  atomicadd(&(node_f.force[0*para.number_of_nodes + node_index[5]]), (delta[5]*delta_j[0]));
  atomicadd(&(node_f.force[1*para.number_of_nodes + node_index[5]]), (delta[5]*delta_j[1]));
  atomicadd(&(node_f.force[2*para.number_of_nodes + node_index[5]]), (delta[5]*delta_j[2]));

  atomicadd(&(node_f.force[0*para.number_of_nodes + node_index[6]]), (delta[6]*delta_j[0]));
  atomicadd(&(node_f.force[1*para.number_of_nodes + node_index[6]]), (delta[6]*delta_j[1]));
  atomicadd(&(node_f.force[2*para.number_of_nodes + node_index[6]]), (delta[6]*delta_j[2]));

  atomicadd(&(node_f.force[0*para.number_of_nodes + node_index[7]]), (delta[7]*delta_j[0]));
  atomicadd(&(node_f.force[1*para.number_of_nodes + node_index[7]]), (delta[7]*delta_j[1]));
  atomicadd(&(node_f.force[2*para.number_of_nodes + node_index[7]]), (delta[7]*delta_j[2]));
}
/*********************************************************/
/** \name System setup and Kernel funktions */
/*********************************************************/
/**kernel to calculate local populations from hydrodynamic fields given by the tcl values.
 * The mapping is given in terms of the equilibrium distribution.
 *
 * Eq. (2.15) Ladd, J. Fluid Mech. 271, 295-309 (1994)
 * Eq. (4) in Berk Usta, Ladd and Butler, JCP 122, 094902 (2005)
 *
 * @param n_a		 Pointer to the lattice site (Input).
*/
#ifndef SHANCHEN
__global__ void calc_n_equilibrium(LB_nodes_gpu n_a) {

  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x;

  if(index<para.number_of_nodes){

    /** default values for fields in lattice units */

    float Rho = para.rho*para.agrid*para.agrid*para.agrid;
    float v[3] = { 0.1f, 0.0f, 0.0f };
    float pi[6] = { Rho*c_sound_sq, 0.0f, Rho*c_sound_sq, 0.0f, 0.0f, Rho*c_sound_sq };

    float rhoc_sq = Rho*c_sound_sq;
    float avg_rho = para.rho*para.agrid*para.agrid*para.agrid;
    float local_rho, local_j[3], *local_pi, trace;

    local_rho  = Rho;

    local_j[0] = Rho * v[0];
    local_j[1] = Rho * v[1];
    local_j[2] = Rho * v[2];

    local_pi = pi;

    /** reduce the pressure tensor to the part needed here */
    local_pi[0] -= rhoc_sq;
    local_pi[2] -= rhoc_sq;
    local_pi[5] -= rhoc_sq;

    trace = local_pi[0] + local_pi[2] + local_pi[5];

    float rho_times_coeff;
    float tmp1,tmp2;

    /** update the q=0 sublattice */
    n_a.vd[0*para.number_of_nodes + index] = 1.f/3.f * (local_rho-avg_rho) - 1.f/2.f*trace;

    /** update the q=1 sublattice */
    rho_times_coeff = 1.f/18.f * (local_rho-avg_rho);

    n_a.vd[1*para.number_of_nodes + index] = rho_times_coeff + 1.f/6.f*local_j[0] + 1.f/4.f*local_pi[0] - 1.f/12.f*trace;
    n_a.vd[2*para.number_of_nodes + index] = rho_times_coeff - 1.f/6.f*local_j[0] + 1.f/4.f*local_pi[0] - 1.f/12.f*trace;
    n_a.vd[3*para.number_of_nodes + index] = rho_times_coeff + 1.f/6.f*local_j[1] + 1.f/4.f*local_pi[2] - 1.f/12.f*trace;
    n_a.vd[4*para.number_of_nodes + index] = rho_times_coeff - 1.f/6.f*local_j[1] + 1.f/4.f*local_pi[2] - 1.f/12.f*trace;
    n_a.vd[5*para.number_of_nodes + index] = rho_times_coeff + 1.f/6.f*local_j[2] + 1.f/4.f*local_pi[5] - 1.f/12.f*trace;
    n_a.vd[6*para.number_of_nodes + index] = rho_times_coeff - 1.f/6.f*local_j[2] + 1.f/4.f*local_pi[5] - 1.f/12.f*trace;
    /** update the q=2 sublattice */
    rho_times_coeff = 1.f/36.f * (local_rho-avg_rho);

    tmp1 = local_pi[0] + local_pi[2];
    tmp2 = 2.0f*local_pi[1];
    n_a.vd[7*para.number_of_nodes + index]  = rho_times_coeff + 1.f/12.f*(local_j[0]+local_j[1]) + 1.f/8.f*(tmp1+tmp2) - 1.f/24.f*trace;
    n_a.vd[8*para.number_of_nodes + index]  = rho_times_coeff - 1.f/12.f*(local_j[0]+local_j[1]) + 1.f/8.f*(tmp1+tmp2) - 1.f/24.f*trace;
    n_a.vd[9*para.number_of_nodes + index]  = rho_times_coeff + 1.f/12.f*(local_j[0]-local_j[1]) + 1.f/8.f*(tmp1-tmp2) - 1.f/24.f*trace;
    n_a.vd[10*para.number_of_nodes + index] = rho_times_coeff - 1.f/12.f*(local_j[0]-local_j[1]) + 1.f/8.f*(tmp1-tmp2) - 1.f/24.f*trace;

    tmp1 = local_pi[0] + local_pi[5];
    tmp2 = 2.0f*local_pi[3];

    n_a.vd[11*para.number_of_nodes + index] = rho_times_coeff + 1.f/12.f*(local_j[0]+local_j[2]) + 1.f/8.f*(tmp1+tmp2) - 1.f/24.f*trace;
    n_a.vd[12*para.number_of_nodes + index] = rho_times_coeff - 1.f/12.f*(local_j[0]+local_j[2]) + 1.f/8.f*(tmp1+tmp2) - 1.f/24.f*trace;
    n_a.vd[13*para.number_of_nodes + index] = rho_times_coeff + 1.f/12.f*(local_j[0]-local_j[2]) + 1.f/8.f*(tmp1-tmp2) - 1.f/24.f*trace;
    n_a.vd[14*para.number_of_nodes + index] = rho_times_coeff - 1.f/12.f*(local_j[0]-local_j[2]) + 1.f/8.f*(tmp1-tmp2) - 1.f/24.f*trace;

    tmp1 = local_pi[2] + local_pi[5];
    tmp2 = 2.0f*local_pi[4];

    n_a.vd[15*para.number_of_nodes + index] = rho_times_coeff + 1.f/12.f*(local_j[1]+local_j[2]) + 1.f/8.f*(tmp1+tmp2) - 1.f/24.f*trace;
    n_a.vd[16*para.number_of_nodes + index] = rho_times_coeff - 1.f/12.f*(local_j[1]+local_j[2]) + 1.f/8.f*(tmp1+tmp2) - 1.f/24.f*trace;
    n_a.vd[17*para.number_of_nodes + index] = rho_times_coeff + 1.f/12.f*(local_j[1]-local_j[2]) + 1.f/8.f*(tmp1-tmp2) - 1.f/24.f*trace;
    n_a.vd[18*para.number_of_nodes + index] = rho_times_coeff - 1.f/12.f*(local_j[1]-local_j[2]) + 1.f/8.f*(tmp1-tmp2) - 1.f/24.f*trace;

    /**set different seed for randomgen on every node */
    n_a.seed[index] = para.your_seed + index;
    //printf("index %i vd: %lf\n", index,n_a.vd[18*para.number_of_nodes + index]);
  }
}
#endif
/** kernel to calculate local populations from hydrodynamic fields
 * from given flow field velocities.  The mapping is given in terms of
 * the equilibrium distribution.
 *
 * Eq. (2.15) Ladd, J. Fluid Mech. 271, 295-309 (1994)
 * Eq. (4) in Berk Usta, Ladd and Butler, JCP 122, 094902 (2005)
 *
 * @param n_a		   the current nodes array (double buffering!)
 * @param single_nodeindex the node to set the velocity for
 * @param velocity         the velocity to set
 */
__global__ void set_u_equilibrium(LB_nodes_gpu n_a, int single_nodeindex,float *velocity) {

  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x;

  if(index == 0){

    /** default values for fields in lattice units */
    float mode[4];
    calc_mode(mode, n_a, single_nodeindex);
    float Rho = mode[0] + para.rho*para.agrid*para.agrid*para.agrid;
    float v[3];
    v[0] = velocity[0];
    v[1] = velocity[1];
    v[2] = velocity[2];

    float pi[6] = { Rho*c_sound_sq, 0.0f, Rho*c_sound_sq, 0.0f, 0.0f, Rho*c_sound_sq };

    float rhoc_sq = Rho*c_sound_sq;
    float avg_rho = para.rho*para.agrid*para.agrid*para.agrid;
    float local_rho, local_j[3], *local_pi, trace;

    local_rho  = Rho;

    local_j[0] = Rho * v[0];
    local_j[1] = Rho * v[1];
    local_j[2] = Rho * v[2];

    local_pi = pi;

    /** reduce the pressure tensor to the part needed here */
    local_pi[0] -= rhoc_sq;
    local_pi[2] -= rhoc_sq;
    local_pi[5] -= rhoc_sq;

    trace = local_pi[0] + local_pi[2] + local_pi[5];

    float rho_times_coeff;
    float tmp1,tmp2;

    /** update the q=0 sublattice */
    n_a.vd[0*para.number_of_nodes + single_nodeindex] = 1.f/3.f * (local_rho-avg_rho) - 1.f/2.f*trace;

    /** update the q=1 sublattice */
    rho_times_coeff = 1.f/18.f * (local_rho-avg_rho);

    n_a.vd[1*para.number_of_nodes + single_nodeindex] = rho_times_coeff + 1.f/6.f*local_j[0] + 1.f/4.f*local_pi[0] - 1.f/12.f*trace;
    n_a.vd[2*para.number_of_nodes + single_nodeindex] = rho_times_coeff - 1.f/6.f*local_j[0] + 1.f/4.f*local_pi[0] - 1.f/12.f*trace;
    n_a.vd[3*para.number_of_nodes + single_nodeindex] = rho_times_coeff + 1.f/6.f*local_j[1] + 1.f/4.f*local_pi[2] - 1.f/12.f*trace;
    n_a.vd[4*para.number_of_nodes + single_nodeindex] = rho_times_coeff - 1.f/6.f*local_j[1] + 1.f/4.f*local_pi[2] - 1.f/12.f*trace;
    n_a.vd[5*para.number_of_nodes + single_nodeindex] = rho_times_coeff + 1.f/6.f*local_j[2] + 1.f/4.f*local_pi[5] - 1.f/12.f*trace;
    n_a.vd[6*para.number_of_nodes + single_nodeindex] = rho_times_coeff - 1.f/6.f*local_j[2] + 1.f/4.f*local_pi[5] - 1.f/12.f*trace;

    /** update the q=2 sublattice */
    rho_times_coeff = 1.f/36.f * (local_rho-avg_rho);

    tmp1 = local_pi[0] + local_pi[2];
    tmp2 = 2.0f*local_pi[1];
    n_a.vd[7*para.number_of_nodes + single_nodeindex]  = rho_times_coeff + 1.f/12.f*(local_j[0]+local_j[1]) + 1.f/8.f*(tmp1+tmp2) - 1.f/24.f*trace;
    n_a.vd[8*para.number_of_nodes + single_nodeindex]  = rho_times_coeff - 1.f/12.f*(local_j[0]+local_j[1]) + 1.f/8.f*(tmp1+tmp2) - 1.f/24.f*trace;
    n_a.vd[9*para.number_of_nodes + single_nodeindex]  = rho_times_coeff + 1.f/12.f*(local_j[0]-local_j[1]) + 1.f/8.f*(tmp1-tmp2) - 1.f/24.f*trace;
    n_a.vd[10*para.number_of_nodes + single_nodeindex] = rho_times_coeff - 1.f/12.f*(local_j[0]-local_j[1]) + 1.f/8.f*(tmp1-tmp2) - 1.f/24.f*trace;

    tmp1 = local_pi[0] + local_pi[5];
    tmp2 = 2.0f*local_pi[3];

    n_a.vd[11*para.number_of_nodes + single_nodeindex] = rho_times_coeff + 1.f/12.f*(local_j[0]+local_j[2]) + 1.f/8.f*(tmp1+tmp2) - 1.f/24.f*trace;
    n_a.vd[12*para.number_of_nodes + single_nodeindex] = rho_times_coeff - 1.f/12.f*(local_j[0]+local_j[2]) + 1.f/8.f*(tmp1+tmp2) - 1.f/24.f*trace;
    n_a.vd[13*para.number_of_nodes + single_nodeindex] = rho_times_coeff + 1.f/12.f*(local_j[0]-local_j[2]) + 1.f/8.f*(tmp1-tmp2) - 1.f/24.f*trace;
    n_a.vd[14*para.number_of_nodes + single_nodeindex] = rho_times_coeff - 1.f/12.f*(local_j[0]-local_j[2]) + 1.f/8.f*(tmp1-tmp2) - 1.f/24.f*trace;

    tmp1 = local_pi[2] + local_pi[5];
    tmp2 = 2.0f*local_pi[4];

    n_a.vd[15*para.number_of_nodes + single_nodeindex] = rho_times_coeff + 1.f/12.f*(local_j[1]+local_j[2]) + 1.f/8.f*(tmp1+tmp2) - 1.f/24.f*trace;
    n_a.vd[16*para.number_of_nodes + single_nodeindex] = rho_times_coeff - 1.f/12.f*(local_j[1]+local_j[2]) + 1.f/8.f*(tmp1+tmp2) - 1.f/24.f*trace;
    n_a.vd[17*para.number_of_nodes + single_nodeindex] = rho_times_coeff + 1.f/12.f*(local_j[1]-local_j[2]) + 1.f/8.f*(tmp1-tmp2) - 1.f/24.f*trace;
    n_a.vd[18*para.number_of_nodes + single_nodeindex] = rho_times_coeff - 1.f/12.f*(local_j[1]-local_j[2]) + 1.f/8.f*(tmp1-tmp2) - 1.f/24.f*trace;

  }

}
/**calculate mass of the hole fluid kernel
 * @param *sum				Pointer to result storage value (Output)
 * @param n_a				Pointer to local node residing in array a (Input)
*/
__global__ void calc_mass(LB_nodes_gpu n_a, float *sum) {
  float mode[4];

  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x;

  if(index<para.number_of_nodes){
    calc_mode(mode, n_a, index);
    float Rho = mode[0] + para.rho*para.agrid*para.agrid*para.agrid;
    //if(n_a.boundary[index]){
      //mode[0] = 0.f;
    //}
    atomicadd(&(sum[0]), Rho);
  }
}
/** (re-)initialization of the node force / set up of external force in lb units
 * @param node_f		Pointer to local node force (Input)
*/
__global__ void reinit_node_force(LB_node_force_gpu node_f){

  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x;

  if(index<para.number_of_nodes){
#ifdef EXTERNAL_FORCES
    if(para.external_force){
      node_f.force[0*para.number_of_nodes + index] = para.ext_force[0]*powf(para.agrid,4)*para.tau*para.tau;
      node_f.force[1*para.number_of_nodes + index] = para.ext_force[1]*powf(para.agrid,4)*para.tau*para.tau;
      node_f.force[2*para.number_of_nodes + index] = para.ext_force[2]*powf(para.agrid,4)*para.tau*para.tau;
    }
    else{
      node_f.force[0*para.number_of_nodes + index] = 0.0f;
      node_f.force[1*para.number_of_nodes + index] = 0.0f;
      node_f.force[2*para.number_of_nodes + index] = 0.0f;
    }
#else
    node_f.force[0*para.number_of_nodes + index] = 0.0f;
    node_f.force[1*para.number_of_nodes + index] = 0.0f;
    node_f.force[2*para.number_of_nodes + index] = 0.0f;
#endif
  }
}


__global__ void init_extern_nodeforces(int n_extern_nodeforces, LB_extern_nodeforce_gpu *extern_nodeforces, LB_node_force_gpu node_f){

  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x;

  if(index<n_extern_nodeforces){
    node_f.force[0*para.number_of_nodes + extern_nodeforces[index].index] = extern_nodeforces[index].force[0]*powf(para.agrid,4)*para.tau*para.tau;
    node_f.force[1*para.number_of_nodes + extern_nodeforces[index].index] = extern_nodeforces[index].force[1]*powf(para.agrid,4)*para.tau*para.tau;
    node_f.force[2*para.number_of_nodes + extern_nodeforces[index].index] = extern_nodeforces[index].force[2]*powf(para.agrid,4)*para.tau*para.tau;
  }
}
#else  //SHANCHEN 

/**calculation of the modes from the velocitydensities (space-transform.)
 * @param n_a		Pointer to local node residing in array a (Input)
 * @param index		node index / thread index (Input)
 * @param mode		Pointer to the local register values mode (Output)
*/
__device__ void calc_m_from_n(LB_nodes_gpu n_a, unsigned int index, float *mode){
  #pragma unroll
  for(int ii=0;ii<SHANCHEN;++ii) { 
  /* mass mode */
  mode[0 + ii * LBQ] = n_a.vd[(0 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(1 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(2 + ii*LBQ ) * para.number_of_nodes + index]
          + n_a.vd[(3 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(4 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(5 + ii*LBQ ) * para.number_of_nodes + index]
          + n_a.vd[(6 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(7 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(8 + ii*LBQ ) * para.number_of_nodes + index]
          + n_a.vd[(9 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(10 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(11 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(12 + ii*LBQ ) * para.number_of_nodes + index]
          + n_a.vd[(13 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(14 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(15 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(16 + ii*LBQ ) * para.number_of_nodes + index]
          + n_a.vd[(17 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(18 + ii*LBQ ) * para.number_of_nodes + index];

  /* momentum modes */
  mode[1 + ii * LBQ] = (n_a.vd[(1 + ii*LBQ ) * para.number_of_nodes + index] - n_a.vd[(2 + ii*LBQ ) * para.number_of_nodes + index]) + (n_a.vd[(7 + ii*LBQ ) * para.number_of_nodes + index] - n_a.vd[(8 + ii*LBQ ) * para.number_of_nodes + index])
          + (n_a.vd[(9 + ii*LBQ ) * para.number_of_nodes + index] - n_a.vd[(10 + ii*LBQ ) * para.number_of_nodes + index]) + (n_a.vd[(11 + ii*LBQ ) * para.number_of_nodes + index] - n_a.vd[(12 + ii*LBQ ) * para.number_of_nodes + index])
          + (n_a.vd[(13 + ii*LBQ ) * para.number_of_nodes + index] - n_a.vd[(14 + ii*LBQ ) * para.number_of_nodes + index]);
  mode[2 + ii * LBQ] = (n_a.vd[(3 + ii*LBQ ) * para.number_of_nodes + index] - n_a.vd[(4 + ii*LBQ ) * para.number_of_nodes + index]) + (n_a.vd[(7 + ii*LBQ ) * para.number_of_nodes + index] - n_a.vd[(8 + ii*LBQ ) * para.number_of_nodes + index])
          - (n_a.vd[(9 + ii*LBQ ) * para.number_of_nodes + index] - n_a.vd[(10 + ii*LBQ ) * para.number_of_nodes + index]) + (n_a.vd[(15 + ii*LBQ ) * para.number_of_nodes + index] - n_a.vd[(16 + ii*LBQ ) * para.number_of_nodes + index])
          + (n_a.vd[(17 + ii*LBQ ) * para.number_of_nodes + index] - n_a.vd[(18 + ii*LBQ ) * para.number_of_nodes + index]);
  mode[3 + ii * LBQ] = (n_a.vd[(5 + ii*LBQ ) * para.number_of_nodes + index] - n_a.vd[(6 + ii*LBQ ) * para.number_of_nodes + index]) + (n_a.vd[(11 + ii*LBQ ) * para.number_of_nodes + index] - n_a.vd[(12 + ii*LBQ ) * para.number_of_nodes + index])
          - (n_a.vd[(13 + ii*LBQ ) * para.number_of_nodes + index] - n_a.vd[(14 + ii*LBQ ) * para.number_of_nodes + index]) + (n_a.vd[(15 + ii*LBQ ) * para.number_of_nodes + index] - n_a.vd[(16 + ii*LBQ ) * para.number_of_nodes + index])
          - (n_a.vd[(17 + ii*LBQ ) * para.number_of_nodes + index] - n_a.vd[(18 + ii*LBQ ) * para.number_of_nodes + index]);

  /* stress modes */
  mode[4 + ii * LBQ] = -(n_a.vd[(0 + ii*LBQ ) * para.number_of_nodes + index]) + n_a.vd[(7 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(8 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(9 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(10 + ii*LBQ ) * para.number_of_nodes + index]
          + n_a.vd[(11 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(12 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(13 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(14 + ii*LBQ ) * para.number_of_nodes + index]
          + n_a.vd[(15 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(16 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(17 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(18 + ii*LBQ ) * para.number_of_nodes + index];
  mode[5 + ii * LBQ] = n_a.vd[(1 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(2 + ii*LBQ ) * para.number_of_nodes + index] - (n_a.vd[(3 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(4 + ii*LBQ ) * para.number_of_nodes + index])
          + (n_a.vd[(11 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(12 + ii*LBQ ) * para.number_of_nodes + index]) + (n_a.vd[(13 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(14 + ii*LBQ ) * para.number_of_nodes + index])
          - (n_a.vd[(15 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(16 + ii*LBQ ) * para.number_of_nodes + index]) - (n_a.vd[(17 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(18 + ii*LBQ ) * para.number_of_nodes + index]);
  mode[6 + ii * LBQ] = (n_a.vd[(1 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(2 + ii*LBQ ) * para.number_of_nodes + index]) + (n_a.vd[(3 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(4 + ii*LBQ ) * para.number_of_nodes + index])
          - (n_a.vd[(11 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(12 + ii*LBQ ) * para.number_of_nodes + index]) - (n_a.vd[(13 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(14 + ii*LBQ ) * para.number_of_nodes + index])
          - (n_a.vd[(15 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(16 + ii*LBQ ) * para.number_of_nodes + index]) - (n_a.vd[(17 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(18 + ii*LBQ ) * para.number_of_nodes + index])
          - 2.f*(n_a.vd[(5 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(6 + ii*LBQ ) * para.number_of_nodes + index] - (n_a.vd[(7 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(8 + ii*LBQ ) * para.number_of_nodes + index])
          - (n_a.vd[(9 + ii*LBQ ) * para.number_of_nodes + index] +n_a.vd[(10 + ii*LBQ ) * para.number_of_nodes + index]));
  mode[7 + ii * LBQ] = n_a.vd[(7 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(8 + ii*LBQ ) * para.number_of_nodes + index] - (n_a.vd[(9 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(10 + ii*LBQ ) * para.number_of_nodes + index]);
  mode[8 + ii * LBQ] = n_a.vd[(11 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(12 + ii*LBQ ) * para.number_of_nodes + index] - (n_a.vd[(13 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(14 + ii*LBQ ) * para.number_of_nodes + index]);
  mode[9 + ii * LBQ] = n_a.vd[(15 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(16 + ii*LBQ ) * para.number_of_nodes + index] - (n_a.vd[(17 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(18 + ii*LBQ ) * para.number_of_nodes + index]);

  /* kinetic modes */
  mode[10 + ii * LBQ] = -2.f*(n_a.vd[(1 + ii*LBQ ) * para.number_of_nodes + index] - n_a.vd[(2 + ii*LBQ ) * para.number_of_nodes + index]) + (n_a.vd[(7 + ii*LBQ ) * para.number_of_nodes + index] - n_a.vd[(8 + ii*LBQ ) * para.number_of_nodes + index])
           + (n_a.vd[(9 + ii*LBQ ) * para.number_of_nodes + index] - n_a.vd[(10 + ii*LBQ ) * para.number_of_nodes + index]) + (n_a.vd[(11 + ii*LBQ ) * para.number_of_nodes + index] - n_a.vd[(12 + ii*LBQ ) * para.number_of_nodes + index])
           + (n_a.vd[(13 + ii*LBQ ) * para.number_of_nodes + index] - n_a.vd[(14 + ii*LBQ ) * para.number_of_nodes + index]);
  mode[11 + ii * LBQ] = -2.f*(n_a.vd[(3 + ii*LBQ ) * para.number_of_nodes + index] - n_a.vd[(4 + ii*LBQ ) * para.number_of_nodes + index]) + (n_a.vd[(7 + ii*LBQ ) * para.number_of_nodes + index] - n_a.vd[(8 + ii*LBQ ) * para.number_of_nodes + index])
           - (n_a.vd[(9 + ii*LBQ ) * para.number_of_nodes + index] - n_a.vd[(10 + ii*LBQ ) * para.number_of_nodes + index]) + (n_a.vd[(15 + ii*LBQ ) * para.number_of_nodes + index] - n_a.vd[(16 + ii*LBQ ) * para.number_of_nodes + index])
           + (n_a.vd[(17 + ii*LBQ ) * para.number_of_nodes + index] - n_a.vd[(18 + ii*LBQ ) * para.number_of_nodes + index]);
  mode[12 + ii * LBQ] = -2.f*(n_a.vd[(5 + ii*LBQ ) * para.number_of_nodes + index] - n_a.vd[(6 + ii*LBQ ) * para.number_of_nodes + index]) + (n_a.vd[(11 + ii*LBQ ) * para.number_of_nodes + index] - n_a.vd[(12 + ii*LBQ ) * para.number_of_nodes + index])
           - (n_a.vd[(13 + ii*LBQ ) * para.number_of_nodes + index] - n_a.vd[(14 + ii*LBQ ) * para.number_of_nodes + index]) + (n_a.vd[(15 + ii*LBQ ) * para.number_of_nodes + index] - n_a.vd[(16 + ii*LBQ ) * para.number_of_nodes + index])
           - (n_a.vd[(17 + ii*LBQ ) * para.number_of_nodes + index] - n_a.vd[(18 + ii*LBQ ) * para.number_of_nodes + index]);
  mode[13 + ii * LBQ] = (n_a.vd[(7 + ii*LBQ ) * para.number_of_nodes + index] - n_a.vd[(8 + ii*LBQ ) * para.number_of_nodes + index]) + (n_a.vd[(9 + ii*LBQ ) * para.number_of_nodes + index] - n_a.vd[(10 + ii*LBQ ) * para.number_of_nodes + index])
           - (n_a.vd[(11 + ii*LBQ ) * para.number_of_nodes + index] - n_a.vd[(12 + ii*LBQ ) * para.number_of_nodes + index]) - (n_a.vd[(13 + ii*LBQ ) * para.number_of_nodes + index] - n_a.vd[(14 + ii*LBQ ) * para.number_of_nodes + index]);
  mode[14 + ii * LBQ] = (n_a.vd[(7 + ii*LBQ ) * para.number_of_nodes + index] - n_a.vd[(8 + ii*LBQ ) * para.number_of_nodes + index]) - (n_a.vd[(9 + ii*LBQ ) * para.number_of_nodes + index] - n_a.vd[(10 + ii*LBQ ) * para.number_of_nodes + index])
           - (n_a.vd[(15 + ii*LBQ ) * para.number_of_nodes + index] - n_a.vd[(16 + ii*LBQ ) * para.number_of_nodes + index]) - (n_a.vd[(17 + ii*LBQ ) * para.number_of_nodes + index] - n_a.vd[(18 + ii*LBQ ) * para.number_of_nodes + index]);
  mode[15 + ii * LBQ] = (n_a.vd[(11 + ii*LBQ ) * para.number_of_nodes + index] - n_a.vd[(12 + ii*LBQ ) * para.number_of_nodes + index]) - (n_a.vd[(13 + ii*LBQ ) * para.number_of_nodes + index] - n_a.vd[(14 + ii*LBQ ) * para.number_of_nodes + index])
           - (n_a.vd[(15 + ii*LBQ ) * para.number_of_nodes + index] - n_a.vd[(16 + ii*LBQ ) * para.number_of_nodes + index]) + (n_a.vd[(17 + ii*LBQ ) * para.number_of_nodes + index] - n_a.vd[(18 + ii*LBQ ) * para.number_of_nodes + index]);
  mode[16 + ii * LBQ] = n_a.vd[(0 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(7 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(8 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(9 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(10 + ii*LBQ ) * para.number_of_nodes + index]
           + n_a.vd[(11 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(12 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(13 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(14 + ii*LBQ ) * para.number_of_nodes + index]
           + n_a.vd[(15 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(16 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(17 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(18 + ii*LBQ ) * para.number_of_nodes + index]
           - 2.f*((n_a.vd[(1 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(2 + ii*LBQ ) * para.number_of_nodes + index]) + (n_a.vd[(3 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(4 + ii*LBQ ) * para.number_of_nodes + index])
           + (n_a.vd[(5 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(6 + ii*LBQ ) * para.number_of_nodes + index]));
  mode[17 + ii * LBQ] = -(n_a.vd[(1 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(2 + ii*LBQ ) * para.number_of_nodes + index]) + (n_a.vd[(3 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(4 + ii*LBQ ) * para.number_of_nodes + index])
           + (n_a.vd[(11 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(12 + ii*LBQ ) * para.number_of_nodes + index]) + (n_a.vd[(13 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(14 + ii*LBQ ) * para.number_of_nodes + index])
           - (n_a.vd[(15 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(16 + ii*LBQ ) * para.number_of_nodes + index]) - (n_a.vd[(17 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(18 + ii*LBQ ) * para.number_of_nodes + index]);
  mode[18 + ii * LBQ] = -(n_a.vd[(1 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(2 + ii*LBQ ) * para.number_of_nodes + index]) - (n_a.vd[(3 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(4 + ii*LBQ ) * para.number_of_nodes + index])
           - (n_a.vd[(11 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(12 + ii*LBQ ) * para.number_of_nodes + index]) - (n_a.vd[(13 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(14 + ii*LBQ ) * para.number_of_nodes + index])
           - (n_a.vd[(15 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(16 + ii*LBQ ) * para.number_of_nodes + index]) - (n_a.vd[(17 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(18 + ii*LBQ ) * para.number_of_nodes + index])
           + 2.f*((n_a.vd[(5 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(6 + ii*LBQ ) * para.number_of_nodes + index]) + (n_a.vd[(7 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(8 + ii*LBQ ) * para.number_of_nodes + index])
           + (n_a.vd[(9 + ii*LBQ ) * para.number_of_nodes + index] + n_a.vd[(10 + ii*LBQ ) * para.number_of_nodes + index]));

 }
}

__device__ void relax_modes(float *mode, unsigned int index, LB_node_force_gpu node_f, LB_values_gpu *d_v){
  float Rho_tot=0.f;
  float u_tot[3]={0.f,0.f,0.f};

  #pragma unroll
  for(int ii=0;ii<SHANCHEN;++ii) { 
      /** re-construct the real density
      * remember that the populations are stored as differences to their
      * equilibrium value */
      d_v[index].rho[ii]= mode[0 + ii * LBQ]+ para.rho[ii]*para.agrid*para.agrid*para.agrid;
      Rho_tot  += mode[0 + ii * LBQ]+ para.rho[ii]*para.agrid*para.agrid*para.agrid;
      u_tot[0] += mode[1 + ii * LBQ];
      u_tot[1] += mode[2 + ii * LBQ];
      u_tot[2] += mode[3 + ii * LBQ];

      /** if forces are present, the momentum density is redefined to
      * inlcude one half-step of the force action.  See the
      * Chapman-Enskog expansion in [Ladd & Verberg]. */

      u_tot[0] += 0.5f*node_f.force[(0+ii*3)*para.number_of_nodes + index];
      u_tot[1] += 0.5f*node_f.force[(1+ii*3)*para.number_of_nodes + index];
      u_tot[2] += 0.5f*node_f.force[(2+ii*3)*para.number_of_nodes + index];

    }
  d_v[index].v[0]=u_tot[0]/Rho_tot; 
  d_v[index].v[1]=u_tot[1]/Rho_tot; 
  d_v[index].v[2]=u_tot[2]/Rho_tot; 

  #pragma unroll
  for(int ii=0;ii<SHANCHEN;++ii) { 
      float Rho; float j[3]; float pi_eq[6];

      Rho = mode[0 + ii * LBQ] + para.rho[ii]*para.agrid*para.agrid*para.agrid ;
      j[0] = Rho * u_tot[0];
      j[1] = Rho * u_tot[1];
      j[2] = Rho * u_tot[2];
      /** equilibrium part of the stress modes (eq13 schiller)*/

      pi_eq[0] = ((j[0]*j[0])+(j[1]*j[1])+(j[2]*j[2]))/Rho;
      pi_eq[1] = ((j[0]*j[0])-(j[1]*j[1]))/Rho;
      pi_eq[2] = (((j[0]*j[0])+(j[1]*j[1])+(j[2]*j[2])) - 3.0f*(j[2]*j[2]))/Rho;
      pi_eq[3] = j[0]*j[1]/Rho;
      pi_eq[4] = j[0]*j[2]/Rho;
      pi_eq[5] = j[1]*j[2]/Rho;
 
      /** in Shan-Chen we have to relax the momentum modes as well using the mobility, but
          the total momentum is conserved */  
      // TODO: implement for SHANCHEN >2 and SHANCHEN == 1 
#if (SHANCHEN > 2 || SHANCHEN == 1 )
#error Not implemented for number of components != 1 
#endif
      mode[1 + ii * LBQ] = j[0] + para.gamma_mobility[0]*(mode[1 + ii * LBQ] - j[0]);
      mode[2 + ii * LBQ] = j[1] + para.gamma_mobility[0]*(mode[2 + ii * LBQ] - j[1]);
      mode[3 + ii * LBQ] = j[2] + para.gamma_mobility[0]*(mode[3 + ii * LBQ] - j[2]);
 
      /** relax the stress modes (eq14 schiller)*/
      mode[4 + ii * LBQ] = pi_eq[0] + para.gamma_bulk[ii]*(mode[4 + ii * LBQ] - pi_eq[0]);
      mode[5 + ii * LBQ] = pi_eq[1] + para.gamma_shear[ii]*(mode[5 + ii * LBQ] - pi_eq[1]);
      mode[6 + ii * LBQ] = pi_eq[2] + para.gamma_shear[ii]*(mode[6 + ii * LBQ] - pi_eq[2]);
      mode[7 + ii * LBQ] = pi_eq[3] + para.gamma_shear[ii]*(mode[7 + ii * LBQ] - pi_eq[3]);
      mode[8 + ii * LBQ] = pi_eq[4] + para.gamma_shear[ii]*(mode[8 + ii * LBQ] - pi_eq[4]);
      mode[9 + ii * LBQ] = pi_eq[5] + para.gamma_shear[ii]*(mode[9 + ii * LBQ] - pi_eq[5]);
    
      /** relax the ghost modes (project them out) */
      /** ghost modes have no equilibrium part due to orthogonality */
      mode[10 + ii * LBQ] = para.gamma_odd[ii]*mode[10 + ii * LBQ];
      mode[11 + ii * LBQ] = para.gamma_odd[ii]*mode[11 + ii * LBQ];
      mode[12 + ii * LBQ] = para.gamma_odd[ii]*mode[12 + ii * LBQ];
      mode[13 + ii * LBQ] = para.gamma_odd[ii]*mode[13 + ii * LBQ];
      mode[14 + ii * LBQ] = para.gamma_odd[ii]*mode[14 + ii * LBQ];
      mode[15 + ii * LBQ] = para.gamma_odd[ii]*mode[15 + ii * LBQ];
      mode[16 + ii * LBQ] = para.gamma_even[ii]*mode[16 + ii * LBQ];
      mode[17 + ii * LBQ] = para.gamma_even[ii]*mode[17 + ii * LBQ];
      mode[18 + ii * LBQ] = para.gamma_even[ii]*mode[18 + ii * LBQ];
 }
}
/**thermalization of the modes with gaussian random numbers
 * @param index		node index / thread index (Input)
 * @param mode		Pointer to the local register values mode (Input/Output)
 * @param *rn		Pointer to randomnumber array of the local node
*/
__device__ void thermalize_modes(float *mode, unsigned int index, LB_randomnr_gpu *rn){
  float Rho;
  float sqrt12=sqrtf(12.f);

  LB_randomnr_gpu rngarray[9];
  #pragma unroll
  for(int ii=0; ii< 9 ; ++ii) { 
#ifdef GAUSSRANDOM
	gaussian_random(&rngarray[ii]);
#else
        random_01(&rngarray[ii]);
	rngarray[ii].randomnr[0]-=0.5f;
	rngarray[ii].randomnr[0]*=sqrt12;
	rngarray[ii].randomnr[1]-=0.5f;
	rngarray[ii].randomnr[1]*=sqrt12;
#endif
  }
  #pragma unroll
  for(int ii=0;ii<SHANCHEN;++ii) { 
      /* SAW: TODO NOTE this works only for 2 components */
      Rho = mode[0 + ii * LBQ] + para.rho[ii]*para.agrid*para.agrid*para.agrid;
      /** momentum modes */
      mode[1 + ii * LBQ] += sqrt((para.mu[ii]*(2.f/3.f)*(1.f-(para.gamma_mobility[0]*para.gamma_mobility[0])))) * (2*ii-1) * rngarray[0].randomnr[0];
      mode[2 + ii * LBQ] += sqrt((para.mu[ii]*(2.f/3.f)*(1.f-(para.gamma_mobility[0]*para.gamma_mobility[0])))) * (2*ii-1) * rngarray[0].randomnr[1];
      mode[3 + ii * LBQ] += sqrt((para.mu[ii]*(2.f/3.f)*(1.f-(para.gamma_mobility[0]*para.gamma_mobility[0])))) * (2*ii-1) * rngarray[1].randomnr[0];
      /** stress modes */
      mode[4 + ii * LBQ] += sqrt(Rho*(para.mu[ii]*(2.f/3.f)*(1.f-(para.gamma_bulk[ii]*para.gamma_bulk[ii])))) * rngarray[1].randomnr[1];
      mode[5 + ii * LBQ] += sqrt(Rho*(para.mu[ii]*(4.f/9.f)*(1.f-(para.gamma_shear[ii]*para.gamma_shear[ii])))) * rngarray[2].randomnr[0];
      mode[6 + ii * LBQ] += sqrt(Rho*(para.mu[ii]*(4.f/3.f)*(1.f-(para.gamma_shear[ii]*para.gamma_shear[ii])))) * rngarray[2].randomnr[1];
      mode[7 + ii * LBQ] += sqrt(Rho*(para.mu[ii]*(1.f/9.f)*(1.f-(para.gamma_shear[ii]*para.gamma_shear[ii])))) * rngarray[3].randomnr[0];
      mode[8 + ii * LBQ] += sqrt(Rho*(para.mu[ii]*(1.f/9.f)*(1.f-(para.gamma_shear[ii]*para.gamma_shear[ii])))) * rngarray[3].randomnr[1];
      mode[9 + ii * LBQ] += sqrt(Rho*(para.mu[ii]*(1.f/9.f)*(1.f-(para.gamma_shear[ii]*para.gamma_shear[ii])))) * rngarray[4].randomnr[0];
      /** ghost modes */
      mode[10 + ii * LBQ] += sqrt(Rho*(para.mu[ii]*(2.f/3.f))) * rngarray[4].randomnr[1];
      mode[11 + ii * LBQ] += sqrt(Rho*(para.mu[ii]*(2.f/3.f))) * rngarray[5].randomnr[0];
      mode[12 + ii * LBQ] += sqrt(Rho*(para.mu[ii]*(2.f/3.f))) * rngarray[5].randomnr[1];
      mode[13 + ii * LBQ] += sqrt(Rho*(para.mu[ii]*(2.f/9.f))) * rngarray[6].randomnr[0];
      mode[14 + ii * LBQ] += sqrt(Rho*(para.mu[ii]*(2.f/9.f))) * rngarray[6].randomnr[1];
      mode[15 + ii * LBQ] += sqrt(Rho*(para.mu[ii]*(2.f/9.f))) * rngarray[7].randomnr[0];
      mode[16 + ii * LBQ] += sqrt(Rho*(para.mu[ii]*(2.f)))     * rngarray[7].randomnr[1];
      mode[17 + ii * LBQ] += sqrt(Rho*(para.mu[ii]*(4.f/9.f))) * rngarray[8].randomnr[0];
      mode[18 + ii * LBQ] += sqrt(Rho*(para.mu[ii]*(4.f/3.f))) * rngarray[8].randomnr[1];
   }
}
/*-------------------------------------------------------*/
/**normalization of the modes need befor backtransformation into velocity space
 * @param mode		Pointer to the local register values mode (Input/Output)
*/
__device__ void normalize_modes(float* mode){
  #pragma unroll
  for(int ii=0;ii<SHANCHEN;++ii) { 

  /** normalization factors enter in the back transformation */
  mode[0 + ii * LBQ] *= 1.f;
  mode[1 + ii * LBQ] *= 3.f;
  mode[2 + ii * LBQ] *= 3.f;
  mode[3 + ii * LBQ] *= 3.f;
  mode[4 + ii * LBQ] *= 3.f/2.f;
  mode[5 + ii * LBQ] *= 9.f/4.f;
  mode[6 + ii * LBQ] *= 3.f/4.f;
  mode[7 + ii * LBQ] *= 9.f;
  mode[8 + ii * LBQ] *= 9.f;
  mode[9 + ii * LBQ] *= 9.f;
  mode[10 + ii * LBQ] *= 3.f/2.f;
  mode[11 + ii * LBQ] *= 3.f/2.f;
  mode[12 + ii * LBQ] *= 3.f/2.f;
  mode[13 + ii * LBQ] *= 9.f/2.f;
  mode[14 + ii * LBQ] *= 9.f/2.f;
  mode[15 + ii * LBQ] *= 9.f/2.f;
  mode[16 + ii * LBQ] *= 1.f/2.f;
  mode[17 + ii * LBQ] *= 9.f/4.f;
  mode[18 + ii * LBQ] *= 3.f/4.f;

}
}
/*-------------------------------------------------------*/
/**backtransformation from modespace to desityspace and streaming with the push method using pbc
 * @param index		node index / thread index (Input)
 * @param mode		Pointer to the local register values mode (Input)
 * @param *n_b		Pointer to local node residing in array b (Output)
*/
__device__ void calc_n_from_modes_push(LB_nodes_gpu n_b, float *mode, unsigned int index){

  unsigned int xyz[3];
  index_to_xyz(index, xyz);
  unsigned int x = xyz[0];
  unsigned int y = xyz[1];
  unsigned int z = xyz[2];

  #pragma unroll
  for(int ii=0;ii<SHANCHEN;++ii) { 
  n_b.vd[(0 + ii*LBQ ) * para.number_of_nodes + x + para.dim_x*y + para.dim_x*para.dim_y*z] = 1.f/3.f * (mode[0 + ii * LBQ] - mode[4 + ii * LBQ] + mode[16 + ii * LBQ]);
  n_b.vd[(1 + ii*LBQ ) * para.number_of_nodes + (x+1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*z] = 1.f/18.f * (mode[0 + ii * LBQ] + mode[1 + ii * LBQ] + mode[5 + ii * LBQ] + mode[6 + ii * LBQ] - mode[17 + ii * LBQ] - mode[18 + ii * LBQ] - 2.f*(mode[10 + ii * LBQ] + mode[16 + ii * LBQ]));
  n_b.vd[(2 + ii*LBQ ) * para.number_of_nodes + (para.dim_x+x-1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*z] = 1.f/18.f * (mode[0 + ii * LBQ] - mode[1 + ii * LBQ] + mode[5 + ii * LBQ] + mode[6 + ii * LBQ] - mode[17 + ii * LBQ] - mode[18 + ii * LBQ] + 2.f*(mode[10 + ii * LBQ] - mode[16 + ii * LBQ]));
  n_b.vd[(3 + ii*LBQ ) * para.number_of_nodes + x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*z] = 1.f/18.f * (mode[0 + ii * LBQ] + mode[2 + ii * LBQ] - mode[5 + ii * LBQ] + mode[6 + ii * LBQ] + mode[17 + ii * LBQ] - mode[18 + ii * LBQ] - 2.f*(mode[11 + ii * LBQ] + mode[16 + ii * LBQ]));
  n_b.vd[(4 + ii*LBQ ) * para.number_of_nodes + x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*z] = 1.f/18.f * (mode[0 + ii * LBQ] - mode[2 + ii * LBQ] - mode[5 + ii * LBQ] + mode[6 + ii * LBQ] + mode[17 + ii * LBQ] - mode[18 + ii * LBQ] + 2.f*(mode[11 + ii * LBQ] - mode[16 + ii * LBQ]));
  n_b.vd[(5 + ii*LBQ ) * para.number_of_nodes + x + para.dim_x*y + para.dim_x*para.dim_y*((z+1)%para.dim_z)] = 1.f/18.f * (mode[0 + ii * LBQ] + mode[3 + ii * LBQ] - 2.f*(mode[6 + ii * LBQ] + mode[12 + ii * LBQ] + mode[16 + ii * LBQ] - mode[18 + ii * LBQ]));
  n_b.vd[(6 + ii*LBQ ) * para.number_of_nodes + x + para.dim_x*y + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z)] = 1.f/18.f * (mode[0 + ii * LBQ] - mode[3 + ii * LBQ] - 2.f*(mode[6 + ii * LBQ] - mode[12 + ii * LBQ] + mode[16 + ii * LBQ] - mode[18 + ii * LBQ]));
  n_b.vd[(7 + ii*LBQ ) * para.number_of_nodes + (x+1)%para.dim_x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*z] = 1.f/36.f * (mode[0 + ii * LBQ] + mode[1 + ii * LBQ] + mode[2 + ii * LBQ] + mode[4 + ii * LBQ] + 2.f*mode[6 + ii * LBQ] + mode[7 + ii * LBQ] + mode[10 + ii * LBQ] + mode[11 + ii * LBQ] + mode[13 + ii * LBQ] + mode[14 + ii * LBQ] + mode[16 + ii * LBQ] + 2.f*mode[18 + ii * LBQ]);
  n_b.vd[(8 + ii*LBQ ) * para.number_of_nodes + (para.dim_x+x-1)%para.dim_x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*z] = 1.f/36.f * (mode[0 + ii * LBQ] - mode[1 + ii * LBQ] - mode[2 + ii * LBQ] + mode[4 + ii * LBQ] + 2.f*mode[6 + ii * LBQ] + mode[7 + ii * LBQ] - mode[10 + ii * LBQ] - mode[11 + ii * LBQ] - mode[13 + ii * LBQ] - mode[14 + ii * LBQ] + mode[16 + ii * LBQ] + 2.f*mode[18 + ii * LBQ]);
  n_b.vd[(9 + ii*LBQ ) * para.number_of_nodes + (x+1)%para.dim_x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*z] = 1.f/36.f * (mode[0 + ii * LBQ] + mode[1 + ii * LBQ] - mode[2 + ii * LBQ] + mode[4 + ii * LBQ] + 2.f*mode[6 + ii * LBQ] - mode[7 + ii * LBQ] + mode[10 + ii * LBQ] - mode[11 + ii * LBQ] + mode[13 + ii * LBQ] - mode[14 + ii * LBQ] + mode[16 + ii * LBQ] + 2.f*mode[18 + ii * LBQ]);
  n_b.vd[(10 + ii*LBQ ) * para.number_of_nodes + (para.dim_x+x-1)%para.dim_x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*z] = 1.f/36.f * (mode[0 + ii * LBQ] - mode[1 + ii * LBQ] + mode[2 + ii * LBQ] + mode[4 + ii * LBQ] + 2.f*mode[6 + ii * LBQ] - mode[7 + ii * LBQ] - mode[10 + ii * LBQ] + mode[11 + ii * LBQ] - mode[13 + ii * LBQ] + mode[14 + ii * LBQ] + mode[16 + ii * LBQ] + 2.f*mode[18 + ii * LBQ]);
  n_b.vd[(11 + ii*LBQ ) * para.number_of_nodes + (x+1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*((z+1)%para.dim_z)] = 1.f/36.f * (mode[0 + ii * LBQ] + mode[1 + ii * LBQ] + mode[3 + ii * LBQ] + mode[4 + ii * LBQ] + mode[5 + ii * LBQ] - mode[6 + ii * LBQ] + mode[8 + ii * LBQ] + mode[10 + ii * LBQ] + mode[12 + ii * LBQ] - mode[13 + ii * LBQ] + mode[15 + ii * LBQ] + mode[16 + ii * LBQ] + mode[17 + ii * LBQ] - mode[18 + ii * LBQ]);
  n_b.vd[(12 + ii*LBQ ) * para.number_of_nodes + (para.dim_x+x-1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z)] = 1.f/36.f * (mode[0 + ii * LBQ] - mode[1 + ii * LBQ] - mode[3 + ii * LBQ] + mode[4 + ii * LBQ] + mode[5 + ii * LBQ] - mode[6 + ii * LBQ] + mode[8 + ii * LBQ] - mode[10 + ii * LBQ] - mode[12 + ii * LBQ] + mode[13 + ii * LBQ] - mode[15 + ii * LBQ] + mode[16 + ii * LBQ] + mode[17 + ii * LBQ] - mode[18 + ii * LBQ]);
  n_b.vd[(13 + ii*LBQ ) * para.number_of_nodes + (x+1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z)] = 1.f/36.f * (mode[0 + ii * LBQ] + mode[1 + ii * LBQ] - mode[3 + ii * LBQ] + mode[4 + ii * LBQ] + mode[5 + ii * LBQ] - mode[6 + ii * LBQ] - mode[8 + ii * LBQ] + mode[10 + ii * LBQ] - mode[12 + ii * LBQ] - mode[13 + ii * LBQ] - mode[15 + ii * LBQ] + mode[16 + ii * LBQ] + mode[17 + ii * LBQ] - mode[18 + ii * LBQ]);
  n_b.vd[(14 + ii*LBQ ) * para.number_of_nodes + (para.dim_x+x-1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*((z+1)%para.dim_z)] = 1.f/36.f * (mode[0 + ii * LBQ] - mode[1 + ii * LBQ] + mode[3 + ii * LBQ] + mode[4 + ii * LBQ] + mode[5 + ii * LBQ] - mode[6 + ii * LBQ] - mode[8 + ii * LBQ] - mode[10 + ii * LBQ] + mode[12 + ii * LBQ] + mode[13 + ii * LBQ] + mode[15 + ii * LBQ] + mode[16 + ii * LBQ] + mode[17 + ii * LBQ] - mode[18 + ii * LBQ]);
  n_b.vd[(15 + ii*LBQ ) * para.number_of_nodes + x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*((z+1)%para.dim_z)] = 1.f/36.f * (mode[0 + ii * LBQ] + mode[2 + ii * LBQ] + mode[3 + ii * LBQ] + mode[4 + ii * LBQ] - mode[5 + ii * LBQ] - mode[6 + ii * LBQ] + mode[9 + ii * LBQ] + mode[11 + ii * LBQ] + mode[12 + ii * LBQ] - mode[14 + ii * LBQ] - mode[15 + ii * LBQ] + mode[16 + ii * LBQ] - mode[17 + ii * LBQ] - mode[18 + ii * LBQ]);
  n_b.vd[(16 + ii*LBQ ) * para.number_of_nodes + x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z)] = 1.f/36.f * (mode[0 + ii * LBQ] - mode[2 + ii * LBQ] - mode[3 + ii * LBQ] + mode[4 + ii * LBQ] - mode[5 + ii * LBQ] - mode[6 + ii * LBQ] + mode[9 + ii * LBQ] - mode[11 + ii * LBQ] - mode[12 + ii * LBQ] + mode[14 + ii * LBQ] + mode[15 + ii * LBQ] + mode[16 + ii * LBQ] - mode[17 + ii * LBQ] - mode[18 + ii * LBQ]);
  n_b.vd[(17 + ii*LBQ ) * para.number_of_nodes + x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z)] = 1.f/36.f * (mode[0 + ii * LBQ] + mode[2 + ii * LBQ] - mode[3 + ii * LBQ] + mode[4 + ii * LBQ] - mode[5 + ii * LBQ] - mode[6 + ii * LBQ] - mode[9 + ii * LBQ] + mode[11 + ii * LBQ] - mode[12 + ii * LBQ] - mode[14 + ii * LBQ] + mode[15 + ii * LBQ] + mode[16 + ii * LBQ] - mode[17 + ii * LBQ] - mode[18 + ii * LBQ]);
  n_b.vd[(18 + ii*LBQ ) * para.number_of_nodes + x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*((z+1)%para.dim_z)] = 1.f/36.f * (mode[0 + ii * LBQ] - mode[2 + ii * LBQ] + mode[3 + ii * LBQ] + mode[4 + ii * LBQ] - mode[5 + ii * LBQ] - mode[6 + ii * LBQ] - mode[9 + ii * LBQ] - mode[11 + ii * LBQ] + mode[12 + ii * LBQ] + mode[14 + ii * LBQ] - mode[15 + ii * LBQ] + mode[16 + ii * LBQ] - mode[17 + ii * LBQ] - mode[18 + ii * LBQ]);

}
}

/** Bounce back boundary conditions.
 * The populations that have propagated into a boundary node
 * are bounced back to the node they came from. This results
 * in no slip boundary conditions.
 *
 * [cf. Ladd and Verberg, J. Stat. Phys. 104(5/6):1191-1251, 2001]
 * @param index			node index / thread index (Input)
 * @param n_b			Pointer to local node residing in array b (Input)
 * @param n_a			Pointer to local node residing in array a (Output) (temp stored in buffer a)
*/
__device__ void bounce_back_read_shanchen(LB_nodes_gpu n_b, LB_nodes_gpu n_a, unsigned int index){
  return ; // SAW TODO
  #pragma unroll
  for(int ii=0;ii<SHANCHEN;++ii) { 
  unsigned int xyz[3];

  if(n_b.boundary[index] == 1){
    index_to_xyz(index, xyz);
    unsigned int x = xyz[0];
    unsigned int y = xyz[1];
    unsigned int z = xyz[2];

    /** store vd temporary in second lattice to avoid race conditions */
    n_a.vd[1*para.number_of_nodes + (x+1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*z] = n_b.vd[2*para.number_of_nodes + index];
    n_a.vd[2*para.number_of_nodes + (para.dim_x+x-1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*z] = n_b.vd[1*para.number_of_nodes + index];
    n_a.vd[3*para.number_of_nodes + x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*z] = n_b.vd[4*para.number_of_nodes + index];
    n_a.vd[4*para.number_of_nodes + x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*z] = n_b.vd[3*para.number_of_nodes + index];
    n_a.vd[5*para.number_of_nodes + x + para.dim_x*y + para.dim_x*para.dim_y*((z+1)%para.dim_z)] = n_b.vd[6*para.number_of_nodes + index];
    n_a.vd[6*para.number_of_nodes + x + para.dim_x*y + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z)] = n_b.vd[5*para.number_of_nodes + index];
    n_a.vd[7*para.number_of_nodes + (x+1)%para.dim_x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*z] = n_b.vd[8*para.number_of_nodes + index];
    n_a.vd[8*para.number_of_nodes + (para.dim_x+x-1)%para.dim_x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*z] = n_b.vd[7*para.number_of_nodes + index];
    n_a.vd[9*para.number_of_nodes + (x+1)%para.dim_x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*z] = n_b.vd[10*para.number_of_nodes + index];
    n_a.vd[10*para.number_of_nodes + (para.dim_x+x-1)%para.dim_x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*z] = n_b.vd[9*para.number_of_nodes + index];
    n_a.vd[11*para.number_of_nodes + (x+1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*((z+1)%para.dim_z)] = n_b.vd[12*para.number_of_nodes + index];
    n_a.vd[12*para.number_of_nodes + (para.dim_x+x-1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z)] = n_b.vd[11*para.number_of_nodes + index]; 
    n_a.vd[13*para.number_of_nodes + (x+1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z)] = n_b.vd[14*para.number_of_nodes + index]; 
    n_a.vd[14*para.number_of_nodes + (para.dim_x+x-1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*((z+1)%para.dim_z)] = n_b.vd[13*para.number_of_nodes + index]; 
    n_a.vd[15*para.number_of_nodes + x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*((z+1)%para.dim_z)] = n_b.vd[16*para.number_of_nodes + index];
    n_a.vd[16*para.number_of_nodes + x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z)] = n_b.vd[15*para.number_of_nodes + index];
    n_a.vd[17*para.number_of_nodes + x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z)] = n_b.vd[18*para.number_of_nodes + index]; 
    n_a.vd[18*para.number_of_nodes + x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*((z+1)%para.dim_z)] = n_b.vd[17*para.number_of_nodes + index];
  }
}
}
/**bounce back read kernel needed to avoid raceconditions
 * @param index			node index / thread index (Input)
 * @param n_b			Pointer to local node residing in array b (Input)
 * @param n_a			Pointer to local node residing in array a (Output) (temp stored in buffer a)
*/
__device__ void bounce_back_write(LB_nodes_gpu n_b, LB_nodes_gpu n_a, unsigned int index){
  return ; // SAW TODO
  #pragma unroll
  for(int ii=0;ii<SHANCHEN;++ii) { 

  unsigned int xyz[3];

  if(n_b.boundary[index] == 1){
    index_to_xyz(index, xyz);
    unsigned int x = xyz[0];
    unsigned int y = xyz[1];
    unsigned int z = xyz[2];

    /** stream vd from boundary node back to origin node */
    n_b.vd[1*para.number_of_nodes + (x+1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*z] = n_a.vd[1*para.number_of_nodes + (x+1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*z]; 
    n_b.vd[2*para.number_of_nodes + (para.dim_x+x-1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*z] = n_a.vd[2*para.number_of_nodes + (para.dim_x+x-1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*z];
    n_b.vd[3*para.number_of_nodes + x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*z] = n_a.vd[3*para.number_of_nodes + x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*z];
    n_b.vd[4*para.number_of_nodes + x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*z] = n_a.vd[4*para.number_of_nodes + x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*z];
    n_b.vd[5*para.number_of_nodes + x + para.dim_x*y + para.dim_x*para.dim_y*((z+1)%para.dim_z)] = n_a.vd[5*para.number_of_nodes + x + para.dim_x*y + para.dim_x*para.dim_y*((z+1)%para.dim_z)];
    n_b.vd[6*para.number_of_nodes + x + para.dim_x*y + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z)] = n_a.vd[6*para.number_of_nodes + x + para.dim_x*y + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z)];
    n_b.vd[7*para.number_of_nodes + (x+1)%para.dim_x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*z] = n_a.vd[7*para.number_of_nodes + (x+1)%para.dim_x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*z];
    n_b.vd[8*para.number_of_nodes + (para.dim_x+x-1)%para.dim_x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*z] = n_a.vd[8*para.number_of_nodes + (para.dim_x+x-1)%para.dim_x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*z];
    n_b.vd[9*para.number_of_nodes + (x+1)%para.dim_x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*z] = n_a.vd[9*para.number_of_nodes + (x+1)%para.dim_x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*z];
    n_b.vd[10*para.number_of_nodes + (para.dim_x+x-1)%para.dim_x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*z] = n_a.vd[10*para.number_of_nodes + (para.dim_x+x-1)%para.dim_x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*z];
    n_b.vd[11*para.number_of_nodes + (x+1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*((z+1)%para.dim_z)] = n_a.vd[11*para.number_of_nodes + (x+1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*((z+1)%para.dim_z)];
    n_b.vd[12*para.number_of_nodes + (para.dim_x+x-1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z)] = n_a.vd[12*para.number_of_nodes + (para.dim_x+x-1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z)];
    n_b.vd[13*para.number_of_nodes + (x+1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z)] = n_a.vd[13*para.number_of_nodes + (x+1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z)];
    n_b.vd[14*para.number_of_nodes + (para.dim_x+x-1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*((z+1)%para.dim_z)] = n_a.vd[14*para.number_of_nodes + (para.dim_x+x-1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*((z+1)%para.dim_z)];
    n_b.vd[15*para.number_of_nodes + x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*((z+1)%para.dim_z)] = n_a.vd[15*para.number_of_nodes + x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*((z+1)%para.dim_z)];
    n_b.vd[16*para.number_of_nodes + x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z)] = n_a.vd[16*para.number_of_nodes + x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z)];
    n_b.vd[17*para.number_of_nodes + x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z)] = n_a.vd[17*para.number_of_nodes + x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z)];
    n_b.vd[18*para.number_of_nodes + x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*((z+1)%para.dim_z)] = n_a.vd[18*para.number_of_nodes + x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*((z+1)%para.dim_z)];
  }
}
}
/** add of (external) forces within the modespace, needed for particle-interaction
 * @param index		node index / thread index (Input)
 * @param mode		Pointer to the local register values mode (Input/Output)
 * @param node_f	Pointer to local node force (Input)
*/
__device__ void apply_forces(unsigned int index, float *mode, LB_node_force_gpu node_f, LB_values_gpu *d_v) {
  float Rho =0.0; 
  float tmpRho;
  float u[3]={0.f,0.f,0.f}, C[6]={0.f,0.f,0.f,0.f,0.f,0.f};

  /* Note: the values d_v were calculated in relax_modes() */

  u[0]=d_v[index].v[0]; 
  u[1]=d_v[index].v[1]; 
  u[2]=d_v[index].v[2]; 


  #pragma unroll
  for(int ii=0;ii<SHANCHEN;++ii) {  //SAW TODO
       C[0] += (1.f + para.gamma_bulk[ii])*u[0]*node_f.force[(0 + ii*3 ) * para.number_of_nodes + index] + 
                1.f/3.f*(para.gamma_bulk[ii]-para.gamma_shear[ii])*(u[0]*node_f.force[(0 + ii*3 ) * para.number_of_nodes + index] + 
                        u[1]*node_f.force[(1 + ii*3 ) * para.number_of_nodes + index] + 
                        u[2]*node_f.force[(2 + ii*3 ) * para.number_of_nodes + index]);
       C[2] += (1.f + para.gamma_bulk[ii])*u[1]*node_f.force[(1 + ii*3 ) * para.number_of_nodes + index] + 
                1.f/3.f*(para.gamma_bulk[ii]-para.gamma_shear[ii])*(u[0]*node_f.force[(0 + ii*3 ) * para.number_of_nodes + index] + 
                        u[1]*node_f.force[(1 + ii*3 ) * para.number_of_nodes + index] + 
                        u[2]*node_f.force[(2 + ii*3 ) * para.number_of_nodes + index]);
       C[5] += (1.f + para.gamma_bulk[ii])*u[2]*node_f.force[(2 + ii*3 ) * para.number_of_nodes + index] + 
                1.f/3.f*(para.gamma_bulk[ii]-para.gamma_shear[ii])*(u[0]*node_f.force[(0 + ii*3 ) * para.number_of_nodes + index] + 
                        u[1]*node_f.force[(1 + ii*3 ) * para.number_of_nodes + index] + 
                        u[2]*node_f.force[(2 + ii*3 ) * para.number_of_nodes + index]);
       C[1] += 1.f/2.f*(1.f+para.gamma_shear[ii])*(u[0]*node_f.force[(1 + ii*3 ) * para.number_of_nodes + index]+
                        u[1]*node_f.force[(0 + ii*3 ) * para.number_of_nodes + index]);
       C[3] += 1.f/2.f*(1.f+para.gamma_shear[ii])*(u[0]*node_f.force[(2 + ii*3 ) * para.number_of_nodes + index]+
                        u[2]*node_f.force[(0 + ii*3 ) * para.number_of_nodes + index]);
       C[4] += 1.f/2.f*(1.f+para.gamma_shear[ii])*(u[1]*node_f.force[(2 + ii*3 ) * para.number_of_nodes + index]+
                        u[2]*node_f.force[(1 + ii*3 ) * para.number_of_nodes + index]);
  }

  #pragma unroll
  for(int ii=0;ii<SHANCHEN;++ii) {  //SAW TODO
      /** update momentum modes */
      mode[1 + ii * LBQ] += 1.f/2.f*(1.f+para.gamma_mobility[0])*node_f.force[(0 + ii*3 ) * para.number_of_nodes + index];
      mode[2 + ii * LBQ] += 1.f/2.f*(1.f+para.gamma_mobility[0])*node_f.force[(1 + ii*3 ) * para.number_of_nodes + index];
      mode[3 + ii * LBQ] += 1.f/2.f*(1.f+para.gamma_mobility[0])*node_f.force[(2 + ii*3 ) * para.number_of_nodes + index];
      	
      /** update stress modes */
      mode[4 + ii * LBQ] += C[0] + C[2] + C[5];
      mode[5 + ii * LBQ] += C[0] - C[2];
      mode[6 + ii * LBQ] += C[0] + C[2] - 2.f*C[5];
      mode[7 + ii * LBQ] += C[1];
      mode[8 + ii * LBQ] += C[3];
      mode[9 + ii * LBQ] += C[4];
    
#ifdef EXTERNAL_FORCES
      if(para.external_force){
        node_f.force[(0 + ii*3 ) * para.number_of_nodes + index] = para.ext_force[0]*powf(para.agrid,4)*para.tau*para.tau;
        node_f.force[(1 + ii*3 ) * para.number_of_nodes + index] = para.ext_force[1]*powf(para.agrid,4)*para.tau*para.tau;
        node_f.force[(2 + ii*3 ) * para.number_of_nodes + index] = para.ext_force[2]*powf(para.agrid,4)*para.tau*para.tau;
      }
      else{
      node_f.force[(0 + ii*3 ) * para.number_of_nodes + index] = 0.f;
      node_f.force[(1 + ii*3 ) * para.number_of_nodes + index] = 0.f;
      node_f.force[(2 + ii*3 ) * para.number_of_nodes + index] = 0.f;
      }
#else
      /** reset force */
      node_f.force[(0 + ii*3 ) * para.number_of_nodes + index] = 0.f;
      node_f.force[(1 + ii*3 ) * para.number_of_nodes + index] = 0.f;
      node_f.force[(2 + ii*3 ) * para.number_of_nodes + index] = 0.f;
#endif
  }
}

/**function used to calc physical values of every node
 * @param index		node index / thread index (Input)
 * @param mode		Pointer to the local register values mode (Input)
 * @param n_a		Pointer to local node residing in array a for boundary flag(Input)
 * @param *d_v		Pointer to local device values (Input/Output)
 * @param singlenode	Flag, if there is only one node
*/
__device__ void calc_values(LB_nodes_gpu n_a, float *mode, LB_values_gpu *d_v, LB_node_force_gpu node_f, unsigned int index, unsigned int singlenode){

  float Rho_tot=0.f;
  float u_tot[3]={0.f,0.f,0.f};
  if(singlenode == 1) index=0;

  if(n_a.boundary[index] != 1){
      #pragma unroll
      for(int ii=0;ii<SHANCHEN;++ii) { 
          /** re-construct the real density
          * remember that the populations are stored as differences to their
          * equilibrium value */
          d_v[index].rho[ii]= mode[0 + ii * 4]+ para.rho[ii]*para.agrid*para.agrid*para.agrid;
          Rho_tot  += mode[0 + ii * 4]+ para.rho[ii]*para.agrid*para.agrid*para.agrid;
          u_tot[0] += mode[1 + ii * 4];
          u_tot[1] += mode[2 + ii * 4];
          u_tot[2] += mode[3 + ii * 4];
    
          /** if forces are present, the momentum density is redefined to
          * inlcude one half-step of the force action.  See the
          * Chapman-Enskog expansion in [Ladd & Verberg]. */
    
          u_tot[0] += 0.5f*node_f.force[(0+ii*3)*para.number_of_nodes + index];
          u_tot[1] += 0.5f*node_f.force[(1+ii*3)*para.number_of_nodes + index];
          u_tot[2] += 0.5f*node_f.force[(2+ii*3)*para.number_of_nodes + index];
      }
      u_tot[0]/=Rho_tot;
      u_tot[1]/=Rho_tot;
      u_tot[2]/=Rho_tot;
    
      d_v[index].v[0]=u_tot[0]; 
      d_v[index].v[1]=u_tot[1]; 
      d_v[index].v[2]=u_tot[2]; 
  } else { 
    #pragma unroll
    for(int ii=0;ii<SHANCHEN;++ii) { 
       d_v[index].rho[ii]   = 1.;
    }
    d_v[index].v[0] = 0.;
    d_v[index].v[1] = 0.; 
    d_v[index].v[2] = 0.; 
  }   
#if 0
  if(singlenode == 1){
    /** equilibrium part of the stress modes */
    /**to print out the stress tensor entries, ensure that in lbgpu.h struct the values are available*/
    d_v[0].pi[0] = ((mode[1 + ii * LBQ]*mode[1 + ii * LBQ]) + (mode[2 + ii * LBQ]*mode[2 + ii * LBQ]) + (mode[3 + ii * LBQ]*mode[3 + ii * LBQ]))/para.rho[ii];
    d_v[0].pi[1] = ((mode[1 + ii * LBQ]*mode[1 + ii * LBQ]) - (mode[2 + ii * LBQ]*mode[2 + ii * LBQ]))/para.rho[ii];
    d_v[0].pi[2] = ((mode[1 + ii * LBQ]*mode[1 + ii * LBQ]) + (mode[2 + ii * LBQ]*mode[2 + ii * LBQ])  + (mode[3 + ii * LBQ]*mode[3 + ii * LBQ])) - 3.0f*(mode[3 + ii * LBQ]*mode[3 + ii * LBQ]))/para.rho[ii];
    d_v[0].pi[3] = mode[1 + ii * LBQ]*mode[2 + ii * LBQ]/para.rho[ii];
    d_v[0].pi[4] = mode[1 + ii * LBQ]*mode[3 + ii * LBQ]/para.rho[ii];
    d_v[0].pi[5] = mode[2 + ii * LBQ]*mode[3 + ii * LBQ]/para.rho[ii];
   } else{
    d_v[index].pi[0] = ((mode[1 + ii * LBQ]*mode[1 + ii * LBQ]) + (mode[2 + ii * LBQ]*mode[2 + ii * LBQ]) + (mode[3 + ii * LBQ]*mode[3 + ii * LBQ]))/para.rho[ii];
    d_v[index].pi[1] = ((mode[1 + ii * LBQ]*mode[1 + ii * LBQ]) - (mode[2 + ii * LBQ]*mode[2 + ii * LBQ]))/para.rho[ii];
    d_v[index].pi[2] = ((mode[1 + ii * LBQ]*mode[1 + ii * LBQ]) + (mode[2 + ii * LBQ]*mode[2 + ii * LBQ])  + (mode[3 + ii * LBQ]*mode[3 + ii * LBQ])) - 3.0f*(mode[3 + ii * LBQ]*mode[3 + ii * LBQ]))/para.rho[ii];
    d_v[index].pi[3] = mode[1 + ii * LBQ]*mode[2 + ii * LBQ]/para.rho[ii];
    d_v[index].pi[4] = mode[1 + ii * LBQ]*mode[3 + ii * LBQ]/para.rho[ii];
    d_v[index].pi[5] = mode[2 + ii * LBQ]*mode[3 + ii * LBQ]/para.rho[ii];
  }
#endif
 }

/** 
 * @param singlenode_index	Single node index        (Input)
 * @param *mode			Pointer to the local register values mode (Output)
 * @param n_a			Pointer to local node residing in array a(Input)
*/
__device__ __inline__ float calc_massmode(LB_nodes_gpu n_a, int single_nodeindex, int component_index){
	
  /** mass mode */
  float mode;
  mode = n_a.vd[(0 + component_index*LBQ ) * para.number_of_nodes + single_nodeindex] + n_a.vd[(1 + component_index*LBQ ) * para.number_of_nodes + single_nodeindex] + n_a.vd[(2 + component_index*LBQ ) * para.number_of_nodes + single_nodeindex] 
          + n_a.vd[(3 + component_index*LBQ ) * para.number_of_nodes + single_nodeindex] + n_a.vd[(4 + component_index*LBQ ) * para.number_of_nodes + single_nodeindex] + n_a.vd[(5 + component_index*LBQ ) * para.number_of_nodes + single_nodeindex]
          + n_a.vd[(6 + component_index*LBQ ) * para.number_of_nodes + single_nodeindex] + n_a.vd[(7 + component_index*LBQ ) * para.number_of_nodes + single_nodeindex] + n_a.vd[(8 + component_index*LBQ ) * para.number_of_nodes + single_nodeindex]
          + n_a.vd[(9 + component_index*LBQ ) * para.number_of_nodes + single_nodeindex] + n_a.vd[(10 + component_index*LBQ ) * para.number_of_nodes + single_nodeindex] + n_a.vd[(11 + component_index*LBQ ) * para.number_of_nodes + single_nodeindex] + n_a.vd[(12 + component_index*LBQ ) * para.number_of_nodes + single_nodeindex]
          + n_a.vd[(13 + component_index*LBQ ) * para.number_of_nodes + single_nodeindex] + n_a.vd[(14 + component_index*LBQ ) * para.number_of_nodes + single_nodeindex] + n_a.vd[(15 + component_index*LBQ ) * para.number_of_nodes + single_nodeindex] + n_a.vd[(16 + component_index*LBQ ) * para.number_of_nodes + single_nodeindex]
          + n_a.vd[(17 + component_index*LBQ ) * para.number_of_nodes + single_nodeindex] + n_a.vd[(18 + component_index*LBQ ) * para.number_of_nodes + single_nodeindex];
 mode += para.rho[component_index]*para.agrid*para.agrid*para.agrid;

 return mode;
}


//SAW TODO: comment
__device__ __inline__ void calc_shanchen_contribution(LB_nodes_gpu n_a,int component_index, int x, int y, int z, float *p){ 
      float tmp_p[3]={0.f,0.f,0.f};
      float pseudo;
      int index;
      index  = (x+1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*z;
      pseudo =  calc_massmode(n_a,index,component_index);
      tmp_p[0]+=pseudo/18.f;

      index  = (para.dim_x+x-1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*z;
      pseudo =  calc_massmode(n_a,index,component_index);
      tmp_p[0]-=pseudo/18.f;

      index  = x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*z;
      pseudo =  calc_massmode(n_a,index,component_index);
      tmp_p[1]+=pseudo/18.f;

      index  = x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*z;
      pseudo =  calc_massmode(n_a,index,component_index);
      tmp_p[1]-=pseudo/18.f;

      index  = x + para.dim_x*y + para.dim_x*para.dim_y*((z+1)%para.dim_z);
      pseudo =  calc_massmode(n_a,index,component_index);
      tmp_p[2]+=pseudo/18.f;	

      index  = x + para.dim_x*y + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z);
      pseudo =  calc_massmode(n_a,index,component_index);
      tmp_p[2]-=pseudo/18.f;

      index  = (x+1)%para.dim_x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*z;
      pseudo =  calc_massmode(n_a,index,component_index);
      tmp_p[0]+=pseudo/36.f;
      tmp_p[1]+=pseudo/36.f;

      index  = (para.dim_x+x-1)%para.dim_x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*z;
      pseudo =  calc_massmode(n_a,index,component_index);
      tmp_p[0]-=pseudo/36.f;
      tmp_p[1]-=pseudo/36.f;

      index  = (x+1)%para.dim_x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*z;
      pseudo =  calc_massmode(n_a,index,component_index);
      tmp_p[0]+=pseudo/36.f;
      tmp_p[1]-=pseudo/36.f;

      index  = (para.dim_x+x-1)%para.dim_x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*z;
      pseudo =  calc_massmode(n_a,index,component_index);
      tmp_p[0]-=pseudo/36.f;
      tmp_p[1]+=pseudo/36.f;

      index  = (x+1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*((z+1)%para.dim_z);
      pseudo =  calc_massmode(n_a,index,component_index);
      tmp_p[0]+=pseudo/36.f;
      tmp_p[2]+=pseudo/36.f;

      index  = (para.dim_x+x-1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z);
      pseudo =  calc_massmode(n_a,index,component_index);
      tmp_p[0]-=pseudo/36.f;
      tmp_p[2]-=pseudo/36.f;

      index  = (x+1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z);
      pseudo =  calc_massmode(n_a,index,component_index);
      tmp_p[0]+=pseudo/36.f;
      tmp_p[2]-=pseudo/36.f;

      index  = (para.dim_x+x-1)%para.dim_x + para.dim_x*y + para.dim_x*para.dim_y*((z+1)%para.dim_z);
      pseudo =  calc_massmode(n_a,index,component_index);
      tmp_p[0]-=pseudo/36.f;
      tmp_p[2]+=pseudo/36.f;

      index  = x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*((z+1)%para.dim_z);
      pseudo =  calc_massmode(n_a,index,component_index);
      tmp_p[1]+=pseudo/36.f;
      tmp_p[2]+=pseudo/36.f;

      index  = x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z);
      pseudo =  calc_massmode(n_a,index,component_index);
      tmp_p[1]-=pseudo/36.f;
      tmp_p[2]-=pseudo/36.f;

      index  = x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*((para.dim_z+z-1)%para.dim_z);
      pseudo =  calc_massmode(n_a,index,component_index);
      tmp_p[1]+=pseudo/36.f;
      tmp_p[2]-=pseudo/36.f;

      index  = x + para.dim_x*((para.dim_y+y-1)%para.dim_y) + para.dim_x*para.dim_y*((z+1)%para.dim_z);
      pseudo =  calc_massmode(n_a,index,component_index);
      tmp_p[1]-=pseudo/36.f;
      tmp_p[2]+=pseudo/36.f;
  
      p[0]=tmp_p[0];
      p[1]=tmp_p[1];
      p[2]=tmp_p[2];
}

/** function to calc shanchen forces // SAW TODO check docs here
 * @param *mode			Pointer to the local register values mode (Output)
 * @param n_a			Pointer to local node residing in array a(Input)
 * @param node_f		Pointer to local node force (Input)
*/
__global__ void lb_shanchen_GPU(LB_nodes_gpu n_a,LB_node_force_gpu node_f){
#ifndef D3Q19
#error Lattices other than D3Q19 not supported
#endif
#if ( SHANCHEN == 1  ) 
  #warning shanchen forces not implemented 
#else  // SHANCHEN == 1 
  
  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int xyz[3];
  float pseudo;

  if(index<para.number_of_nodes){
     /*Let's first identify the neighboring nodes */
     index_to_xyz(index, xyz);
     int x = xyz[0];
     int y = xyz[1];
     int z = xyz[2];
     
     #pragma unroll
     for(int ii=0;ii<SHANCHEN;ii++){ 
       float p[3]={0.f,0.f,0.f};
       pseudo =  calc_massmode(n_a,index,ii);
       #pragma unroll
       for(int jj=0;jj<SHANCHEN;jj++){ 
             float tmpp[3]={0.f,0.f,0.f};
             calc_shanchen_contribution(n_a, jj, x,y,z, tmpp);
// SAW: TODO  coupling HAS to be rescaled with agrid....
             p[0] += - para.coupling[(SHANCHEN)*ii+jj]  * pseudo  * tmpp[0];
             p[1] += - para.coupling[(SHANCHEN)*ii+jj]  * pseudo  * tmpp[1];
             p[2] += - para.coupling[(SHANCHEN)*ii+jj]  * pseudo  * tmpp[2];
       }
       node_f.force[(0+ii*3)*para.number_of_nodes + index]+=p[0];
       node_f.force[(1+ii*3)*para.number_of_nodes + index]+=p[1];
       node_f.force[(2+ii*3)*para.number_of_nodes + index]+=p[2];
     }
  }
#endif // SHANCHEN == 1   // SAW TODO: finish implementing
  return; 
}

//SAW TODO: comment
void lb_calc_shanchen_GPU(){
  /** values for the kernel call */
  int threads_per_block = 64;
  int blocks_per_grid_y = 4;
  int blocks_per_grid_x = (lbpar_gpu.number_of_nodes + threads_per_block * blocks_per_grid_y - 1) /(threads_per_block * blocks_per_grid_y);
  dim3 dim_grid = make_uint3(blocks_per_grid_x, blocks_per_grid_y, 1);

  KERNELCALL(lb_shanchen_GPU, dim_grid, threads_per_block,(*current_nodes, node_f));

}


/** 
 * @param node_index	node index around (8) particle (Input)
 * @param *mode			Pointer to the local register values mode (Output)
 * @param n_a			Pointer to local node residing in array a(Input)
*/
__device__ void calc_mode(float *mode, LB_nodes_gpu n_a, unsigned int node_index, int component_index){
	
  /** mass mode */
  mode[0] = n_a.vd[(0 + component_index*LBQ ) * para.number_of_nodes + node_index] + n_a.vd[(1 + component_index*LBQ ) * para.number_of_nodes + node_index] + n_a.vd[(2 + component_index*LBQ ) * para.number_of_nodes + node_index] 
          + n_a.vd[(3 + component_index*LBQ ) * para.number_of_nodes + node_index] + n_a.vd[(4 + component_index*LBQ ) * para.number_of_nodes + node_index] + n_a.vd[(5 + component_index*LBQ ) * para.number_of_nodes + node_index]
          + n_a.vd[(6 + component_index*LBQ ) * para.number_of_nodes + node_index] + n_a.vd[(7 + component_index*LBQ ) * para.number_of_nodes + node_index] + n_a.vd[(8 + component_index*LBQ ) * para.number_of_nodes + node_index]
          + n_a.vd[(9 + component_index*LBQ ) * para.number_of_nodes + node_index] + n_a.vd[(10 + component_index*LBQ ) * para.number_of_nodes + node_index] + n_a.vd[(11 + component_index*LBQ ) * para.number_of_nodes + node_index] + n_a.vd[(12 + component_index*LBQ ) * para.number_of_nodes + node_index]
          + n_a.vd[(13 + component_index*LBQ ) * para.number_of_nodes + node_index] + n_a.vd[(14 + component_index*LBQ ) * para.number_of_nodes + node_index] + n_a.vd[(15 + component_index*LBQ ) * para.number_of_nodes + node_index] + n_a.vd[(16 + component_index*LBQ ) * para.number_of_nodes + node_index]
          + n_a.vd[(17 + component_index*LBQ ) * para.number_of_nodes + node_index] + n_a.vd[(18 + component_index*LBQ ) * para.number_of_nodes + node_index];

  /** momentum modes */
  mode[1] = (n_a.vd[(1 + component_index*LBQ ) * para.number_of_nodes + node_index] - n_a.vd[(2 + component_index*LBQ ) * para.number_of_nodes + node_index]) + (n_a.vd[(7 + component_index*LBQ ) * para.number_of_nodes + node_index] - n_a.vd[(8 + component_index*LBQ ) * para.number_of_nodes + node_index])
          + (n_a.vd[(9 + component_index*LBQ ) * para.number_of_nodes + node_index] - n_a.vd[(10 + component_index*LBQ ) * para.number_of_nodes + node_index]) + (n_a.vd[(11 + component_index*LBQ ) * para.number_of_nodes + node_index] - n_a.vd[(12 + component_index*LBQ ) * para.number_of_nodes + node_index])
          + (n_a.vd[(13 + component_index*LBQ ) * para.number_of_nodes + node_index] - n_a.vd[(14 + component_index*LBQ ) * para.number_of_nodes + node_index]);
  mode[2] = (n_a.vd[(3 + component_index*LBQ ) * para.number_of_nodes + node_index] - n_a.vd[(4 + component_index*LBQ ) * para.number_of_nodes + node_index]) + (n_a.vd[(7 + component_index*LBQ ) * para.number_of_nodes + node_index] - n_a.vd[(8 + component_index*LBQ ) * para.number_of_nodes + node_index])
          - (n_a.vd[(9 + component_index*LBQ ) * para.number_of_nodes + node_index] - n_a.vd[(10 + component_index*LBQ ) * para.number_of_nodes + node_index]) + (n_a.vd[(15 + component_index*LBQ ) * para.number_of_nodes + node_index] - n_a.vd[(16 + component_index*LBQ ) * para.number_of_nodes + node_index])
          + (n_a.vd[(17 + component_index*LBQ ) * para.number_of_nodes + node_index] - n_a.vd[(18 + component_index*LBQ ) * para.number_of_nodes + node_index]);
  mode[3] = (n_a.vd[(5 + component_index*LBQ ) * para.number_of_nodes + node_index] - n_a.vd[(6 + component_index*LBQ ) * para.number_of_nodes + node_index]) + (n_a.vd[(11 + component_index*LBQ ) * para.number_of_nodes + node_index] - n_a.vd[(12 + component_index*LBQ ) * para.number_of_nodes + node_index])
          - (n_a.vd[(13 + component_index*LBQ ) * para.number_of_nodes + node_index] - n_a.vd[(14 + component_index*LBQ ) * para.number_of_nodes + node_index]) + (n_a.vd[(15 + component_index*LBQ ) * para.number_of_nodes + node_index] - n_a.vd[(16 + component_index*LBQ ) * para.number_of_nodes + node_index])
          - (n_a.vd[(17 + component_index*LBQ ) * para.number_of_nodes + node_index] - n_a.vd[(18 + component_index*LBQ ) * para.number_of_nodes + node_index]);
}
#ifdef SHANCHEN
/**calculate temperature of the fluid kernel
 * @param *cpu_jsquared			Pointer to result storage value (Output)
 * @param n_a				Pointer to local node residing in array a (Input)
*/
__global__ void temperature(LB_nodes_gpu n_a, float *cpu_jsquared) {
  float mode[4];
  float jsquared = 0.f;
  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x;

  if(index<para.number_of_nodes){
    if(!n_a.boundary[index]){
     #pragma unroll
     for(int ii=0;ii<SHANCHEN;++ii) {  //SAW TODO: check that temperature is computed correctly in SC
         calc_mode(mode, n_a, index,ii);
         jsquared = mode[1]*mode[1]+mode[2]*mode[2]+mode[3]*mode[3];
         atomicadd(cpu_jsquared, jsquared);
     }
   }
 }
}
#endif

/*********************************************************/
/** \name Coupling part */
/*********************************************************/
/**(Eq. (12) Ahlrichs and Duenweg, JCP 111(17):8225 (1999))
 * @param n_a			Pointer to local node residing in array a (Input)
 * @param *delta		Pointer for the weighting of particle position (Output)
 * @param *delta_j		Pointer for the weighting of particle momentum (Output)
 * @param *particle_data	Pointer to the particle position and velocity (Input)
 * @param *particle_force	Pointer to the particle force (Input)
 * @param part_index		particle id / thread id (Input)
 * @param *rn_part		Pointer to randomnumber array of the particle
 * @param node_index		node index around (8) particle (Output)
*/
#ifdef SHANCHEN
__device__ void calc_viscous_force(LB_nodes_gpu n_a, float *delta, float * partgrad1, float * partgrad2, float * partgrad3, LB_particle_gpu *particle_data, LB_particle_force_gpu *particle_force, unsigned int part_index, LB_randomnr_gpu *rn_part, float *delta_j, unsigned int *node_index,LB_values_gpu *d_v){
	
 float mode[4];
 int my_left[3];
 float interpolated_u1, interpolated_u2, interpolated_u3;
 float interpolated_rho[SHANCHEN];
 float gradrho1, gradrho2, gradrho3;
 float Rho;
 float temp_delta[6];
 float temp_delta_half[6];
 float value;
 float tmpforce[3*SHANCHEN];
 float viscforce[3*SHANCHEN];

 #pragma unroll
 for(int ii=0; ii<SHANCHEN; ++ii){ 
   #pragma unroll
   for(int jj=0; jj<3; ++jj){ 
    tmpforce[jj+ii*3]=0.;
    viscforce[jj+ii*3]=0.;
    delta_j[jj+ii*3]=0.f;
   }
   #pragma unroll
   for(int jj=0; jj<8; ++jj){ 
    partgrad1[jj+ii*8]=0.;
    partgrad2[jj+ii*8]=0.;
    partgrad3[jj+ii*8]=0.;
   }
 }
 /** see ahlrichs + duenweg page 8227 equ (10) and (11) */
 #pragma unroll
 for(int i=0; i<3; ++i){
   float scaledpos = particle_data[part_index].p[i]/para.agrid - 0.5f;
   my_left[i] = (int)(floorf(scaledpos));
   //printf("scaledpos %f \t myleft: %d \n", scaledpos, my_left[i]);
   temp_delta[3+i] = scaledpos - my_left[i];
   temp_delta[i] = 1.f - temp_delta[3+i];
   /**further value used for interpolation of fluid velocity at part pos near boundaries */
   temp_delta_half[3+i] = (scaledpos - my_left[i])*2.f;
   temp_delta_half[i] = 2.f - temp_delta_half[3+i];
 }

 delta[0] = temp_delta[0] * temp_delta[1] * temp_delta[2];
 delta[1] = temp_delta[3] * temp_delta[1] * temp_delta[2];
 delta[2] = temp_delta[0] * temp_delta[4] * temp_delta[2];
 delta[3] = temp_delta[3] * temp_delta[4] * temp_delta[2];
 delta[4] = temp_delta[0] * temp_delta[1] * temp_delta[5];
 delta[5] = temp_delta[3] * temp_delta[1] * temp_delta[5];
 delta[6] = temp_delta[0] * temp_delta[4] * temp_delta[5];
 delta[7] = temp_delta[3] * temp_delta[4] * temp_delta[5];

 // modulo for negative numbers is strange at best, shift to make sure we are positive
 int x = my_left[0] + para.dim_x;
 int y = my_left[1] + para.dim_y;
 int z = my_left[2] + para.dim_z;

 node_index[0] = x%para.dim_x     + para.dim_x*(y%para.dim_y)     + para.dim_x*para.dim_y*(z%para.dim_z);
 node_index[1] = (x+1)%para.dim_x + para.dim_x*(y%para.dim_y)     + para.dim_x*para.dim_y*(z%para.dim_z);
 node_index[2] = x%para.dim_x     + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*(z%para.dim_z);
 node_index[3] = (x+1)%para.dim_x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*(z%para.dim_z);
 node_index[4] = x%para.dim_x     + para.dim_x*(y%para.dim_y)     + para.dim_x*para.dim_y*((z+1)%para.dim_z);
 node_index[5] = (x+1)%para.dim_x + para.dim_x*(y%para.dim_y)     + para.dim_x*para.dim_y*((z+1)%para.dim_z);
 node_index[6] = x%para.dim_x     + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*((z+1)%para.dim_z);
 node_index[7] = (x+1)%para.dim_x + para.dim_x*((y+1)%para.dim_y) + para.dim_x*para.dim_y*((z+1)%para.dim_z);

 particle_force[part_index].f[0] = 0.f;
 particle_force[part_index].f[1] = 0.f;
 particle_force[part_index].f[2] = 0.f;

 interpolated_u1 = interpolated_u2 = interpolated_u3 = 0.f;
 #pragma unroll
 for(int i=0; i<8; ++i){ 
  interpolated_u1 += d_v[node_index[i]].v[0]/8.;  
  interpolated_u2 += d_v[node_index[i]].v[1]/8.;
  interpolated_u3 += d_v[node_index[i]].v[2]/8.;
 }

 /* Shan-Chen-like part */
 #pragma unroll
 for(int ii=0; ii<SHANCHEN; ++ii){ 
  float solvation2 = particle_data[part_index].solvation[2*ii + 1];
                    // delta[0]*delta[1]*delta[2]*delta[3]*delta[4]*delta[5]*
                    // delta[6]*delta[7]*particle_data[part_index].solvation[2*ii + 1]*256.f ; 
   
  interpolated_rho[ii]  = 0.f;
  gradrho1 = gradrho2 = gradrho3 = 0.f;
  
 // SAW TODO: introduce the density dependence in friction
 // SAW TODO comment on the gradient calculation...
  calc_mode(mode, n_a, node_index[0],ii);
  Rho = mode[0] + para.rho[ii]*para.agrid*para.agrid*para.agrid;
  interpolated_rho[ii] += delta[0] * Rho; 
  partgrad1[ii*8 + 0] += Rho * solvation2;
  partgrad2[ii*8 + 0] += Rho * solvation2;
  partgrad3[ii*8 + 0] += Rho * solvation2;
  // SAW TODO check the weighted grad coefficients...
  gradrho1 -=(delta[0] + delta[1]) * Rho; 
  gradrho2 -=(delta[0] + delta[2]) * Rho; 
  gradrho3 -=(delta[0] + delta[4]) * Rho; 

  calc_mode(mode, n_a, node_index[1],ii); 
  Rho = mode[0] +  para.rho[ii]*para.agrid*para.agrid*para.agrid; 
  interpolated_rho[ii] += delta[1] * Rho; 
  partgrad1[ii*8 + 1] -= Rho * solvation2;
  partgrad2[ii*8 + 1] += Rho * solvation2;
  partgrad3[ii*8 + 1] += Rho * solvation2;
  gradrho1 +=(delta[1] + delta[0]) * Rho; 
  gradrho2 -=(delta[1] + delta[3]) * Rho; 
  gradrho3 -=(delta[1] + delta[5]) * Rho; 
  
  calc_mode(mode, n_a, node_index[2],ii);
  Rho = mode[0] + para.rho[ii]*para.agrid*para.agrid*para.agrid;	
  interpolated_rho[ii] += delta[2] * Rho; 
  partgrad1[ii*8 + 2] += Rho * solvation2;
  partgrad2[ii*8 + 2] -= Rho * solvation2;
  partgrad3[ii*8 + 2] += Rho * solvation2;
  gradrho1 -=(delta[2] + delta[3]) * Rho; 
  gradrho2 +=(delta[2] + delta[0]) * Rho; 
  gradrho3 -=(delta[2] + delta[6]) * Rho; 

  calc_mode(mode, n_a, node_index[3],ii);
  Rho = mode[0] + para.rho[ii]*para.agrid*para.agrid*para.agrid;	
  interpolated_rho[ii] += delta[3] * Rho; 
  partgrad1[ii*8 + 3] -= Rho * solvation2;
  partgrad2[ii*8 + 3] -= Rho * solvation2;
  partgrad3[ii*8 + 3] += Rho * solvation2;
  gradrho1 +=(delta[3] + delta[2]) * Rho; 
  gradrho2 +=(delta[3] + delta[1]) * Rho; 
  gradrho3 -=(delta[3] + delta[7]) * Rho; 

  calc_mode(mode, n_a, node_index[4],ii);
  Rho = mode[0] + para.rho[ii]*para.agrid*para.agrid*para.agrid;	
  interpolated_rho[ii] += delta[4] * Rho; 
  partgrad1[ii*8 + 4] += Rho * solvation2;
  partgrad2[ii*8 + 4] += Rho * solvation2;
  partgrad3[ii*8 + 4] -= Rho * solvation2;
  gradrho1 -=(delta[4] + delta[5]) * Rho; 
  gradrho2 -=(delta[4] + delta[6]) * Rho; 
  gradrho3 +=(delta[4] + delta[0]) * Rho; 

  calc_mode(mode, n_a, node_index[5],ii);
  Rho = mode[0] + para.rho[ii]*para.agrid*para.agrid*para.agrid;	
  interpolated_rho[ii] += delta[5] * Rho; 
  partgrad1[ii*8 + 5] -= Rho * solvation2;
  partgrad2[ii*8 + 5] += Rho * solvation2;
  partgrad3[ii*8 + 5] -= Rho * solvation2;
  gradrho1 +=(delta[5] + delta[4]) * Rho; 
  gradrho2 -=(delta[5] + delta[7]) * Rho; 
  gradrho3 +=(delta[5] + delta[1]) * Rho; 

  calc_mode(mode, n_a, node_index[6],ii);
  Rho = mode[0] + para.rho[ii]*para.agrid*para.agrid*para.agrid;	
  interpolated_rho[ii] += delta[6] * Rho; 
  partgrad1[ii*8 + 6] += Rho * solvation2;
  partgrad2[ii*8 + 6] -= Rho * solvation2;
  partgrad3[ii*8 + 6] -= Rho * solvation2;
  gradrho1 -=(delta[6] + delta[7]) * Rho; 
  gradrho2 +=(delta[6] + delta[4]) * Rho; 
  gradrho3 +=(delta[6] + delta[2]) * Rho; 

  calc_mode(mode, n_a, node_index[7],ii);
  Rho = mode[0] + para.rho[ii]*para.agrid*para.agrid*para.agrid;	
  interpolated_rho[ii] += delta[7] * Rho; 
  partgrad1[ii*8 + 7] -= Rho * solvation2;
  partgrad2[ii*8 + 7] -= Rho * solvation2;
  partgrad3[ii*8 + 7] -= Rho * solvation2;
  gradrho1 +=(delta[7] + delta[6]) * Rho; 
  gradrho2 +=(delta[7] + delta[5]) * Rho; 
  gradrho3 +=(delta[7] + delta[3]) * Rho; 

  /* normalize the gradient to md units TODO: is that correct?*/
  gradrho1 *= para.agrid; 
  gradrho2 *= para.agrid; 
  gradrho3 *= para.agrid; 

  tmpforce[0+ii*3] += particle_data[part_index].solvation[2*ii] * gradrho1 ; 
  tmpforce[1+ii*3] += particle_data[part_index].solvation[2*ii] * gradrho2 ;
  tmpforce[2+ii*3] += particle_data[part_index].solvation[2*ii] * gradrho3 ;

  particle_force[part_index].f[0] += tmpforce[0+ii*3];
  particle_force[part_index].f[1] += tmpforce[1+ii*3];
  particle_force[part_index].f[2] += tmpforce[2+ii*3];
 }

  /** calculate viscous force
   * take care to rescale velocities with time_step and transform to MD units
   * (Eq. (9) Ahlrichs and Duenweg, JCP 111(17):8225 (1999)) */
 float rhotot=0;
 
 #pragma unroll
 for(int ii=0; ii<SHANCHEN; ++ii){ 
	rhotot+=interpolated_rho[ii];
 }

 /* Viscous force */
 #pragma unroll
 for(int ii=0; ii<SHANCHEN; ++ii){ 
#ifdef LB_ELECTROHYDRODYNAMICS
  viscforce[0+ii*3] += - interpolated_rho[ii]*para.friction[ii] * (particle_data[part_index].v[0]/para.time_step - interpolated_u1*para.agrid/para.tau - particle_data[part_index].mu_E[0])/rhotot;
  viscforce[1+ii*3] += - interpolated_rho[ii]*para.friction[ii] * (particle_data[part_index].v[1]/para.time_step - interpolated_u2*para.agrid/para.tau - particle_data[part_index].mu_E[1])/rhotot;
  viscforce[2+ii*3] += - interpolated_rho[ii]*para.friction[ii] * (particle_data[part_index].v[2]/para.time_step - interpolated_u3*para.agrid/para.tau - particle_data[part_index].mu_E[2])/rhotot;
#else
  viscforce[0+ii*3] += - interpolated_rho[ii]*para.friction[ii] * (particle_data[part_index].v[0]/para.time_step - interpolated_u1*para.agrid/para.tau)/rhotot;
  viscforce[1+ii*3] += - interpolated_rho[ii]*para.friction[ii] * (particle_data[part_index].v[1]/para.time_step - interpolated_u2*para.agrid/para.tau)/rhotot;
  viscforce[2+ii*3] += - interpolated_rho[ii]*para.friction[ii] * (particle_data[part_index].v[2]/para.time_step - interpolated_u3*para.agrid/para.tau)/rhotot;
#endif
  /** add stochastic force of zero mean (Ahlrichs, Duenweg equ. 15)*/
#ifdef GAUSSRANDOM
  gaussian_random(rn_part);
  viscforce[0+ii*3] += para.lb_coupl_pref2[ii]*rn_part->randomnr[0];
  viscforce[1+ii*3] += para.lb_coupl_pref2[ii]*rn_part->randomnr[1];
  gaussian_random(rn_part);
  viscforce[2+ii*3] += para.lb_coupl_pref2[ii]*rn_part->randomnr[0];
#else
  random_01(rn_part);
  viscforce[0+ii*3] += para.lb_coupl_pref[ii]*(rn_part->randomnr[0]-0.5f);
  viscforce[1+ii*3] += para.lb_coupl_pref[ii]*(rn_part->randomnr[1]-0.5f);
  random_01(rn_part);
  viscforce[2+ii*3] += para.lb_coupl_pref[ii]*(rn_part->randomnr[0]-0.5f);
#endif	  
  /** delta_j for transform momentum transfer to lattice units which is done in calc_node_force
  (Eq. (12) Ahlrichs and Duenweg, JCP 111(17):8225 (1999)) */


  particle_force[part_index].f[0] += viscforce[0+ii*3];
  particle_force[part_index].f[1] += viscforce[1+ii*3];
  particle_force[part_index].f[2] += viscforce[2+ii*3];

  delta_j[0+3*ii] -=  (tmpforce[0+ii*3]+viscforce[0+ii*3])*para.time_step*para.tau/para.agrid;
  delta_j[1+3*ii] -=  (tmpforce[1+ii*3]+viscforce[1+ii*3])*para.time_step*para.tau/para.agrid;
  delta_j[2+3*ii] -=  (tmpforce[2+ii*3]+viscforce[2+ii*3])*para.time_step*para.tau/para.agrid;  	
 }
}
#endif

/**calcutlation of the node force caused by the particles, with atomicadd due to avoiding race conditions 
	(Eq. (14) Ahlrichs and Duenweg, JCP 111(17):8225 (1999))
 * @param *delta		Pointer for the weighting of particle position (Input)
 * @param *delta_j		Pointer for the weighting of particle momentum (Input)
 * @param node_index		node index around (8) particle (Input)
 * @param node_f    		Pointer to the node force (Output).
*/
__device__ void calc_node_force(float *delta, float *delta_j, float * partgrad1, float * partgrad2, float * partgrad3,  unsigned int *node_index, LB_node_force_gpu node_f){
/* SAW TODO: should the drag depend on the density?? */
 #pragma unroll
 for(int ii=0; ii < SHANCHEN ; ++ii) { 
  atomicadd(&(node_f.force[(0+ii*3)*para.number_of_nodes + node_index[0]]), (delta[0]*delta_j[0+ii*3] + partgrad1[ii*8+0]));
  atomicadd(&(node_f.force[(1+ii*3)*para.number_of_nodes + node_index[0]]), (delta[0]*delta_j[1+ii*3] + partgrad2[ii*8+0]));
  atomicadd(&(node_f.force[(2+ii*3)*para.number_of_nodes + node_index[0]]), (delta[0]*delta_j[2+ii*3] + partgrad3[ii*8+0]));
                                                                                                    
  atomicadd(&(node_f.force[(0+ii*3)*para.number_of_nodes + node_index[1]]), (delta[1]*delta_j[0+ii*3] + partgrad1[ii*8+1]));
  atomicadd(&(node_f.force[(1+ii*3)*para.number_of_nodes + node_index[1]]), (delta[1]*delta_j[1+ii*3] + partgrad2[ii*8+1]));
  atomicadd(&(node_f.force[(2+ii*3)*para.number_of_nodes + node_index[1]]), (delta[1]*delta_j[2+ii*3] + partgrad3[ii*8+1]));
                                                                                                    
  atomicadd(&(node_f.force[(0+ii*3)*para.number_of_nodes + node_index[2]]), (delta[2]*delta_j[0+ii*3] + partgrad1[ii*8+2]));
  atomicadd(&(node_f.force[(1+ii*3)*para.number_of_nodes + node_index[2]]), (delta[2]*delta_j[1+ii*3] + partgrad2[ii*8+2]));
  atomicadd(&(node_f.force[(2+ii*3)*para.number_of_nodes + node_index[2]]), (delta[2]*delta_j[2+ii*3] + partgrad3[ii*8+2]));
                                                                                                    
  atomicadd(&(node_f.force[(0+ii*3)*para.number_of_nodes + node_index[3]]), (delta[3]*delta_j[0+ii*3] + partgrad1[ii*8+3]));
  atomicadd(&(node_f.force[(1+ii*3)*para.number_of_nodes + node_index[3]]), (delta[3]*delta_j[1+ii*3] + partgrad2[ii*8+3]));
  atomicadd(&(node_f.force[(2+ii*3)*para.number_of_nodes + node_index[3]]), (delta[3]*delta_j[2+ii*3] + partgrad3[ii*8+3]));
                                                                                                    
  atomicadd(&(node_f.force[(0+ii*3)*para.number_of_nodes + node_index[4]]), (delta[4]*delta_j[0+ii*3] + partgrad1[ii*8+4]));
  atomicadd(&(node_f.force[(1+ii*3)*para.number_of_nodes + node_index[4]]), (delta[4]*delta_j[1+ii*3] + partgrad2[ii*8+4]));
  atomicadd(&(node_f.force[(2+ii*3)*para.number_of_nodes + node_index[4]]), (delta[4]*delta_j[2+ii*3] + partgrad3[ii*8+4]));
                                                                                                    
  atomicadd(&(node_f.force[(0+ii*3)*para.number_of_nodes + node_index[5]]), (delta[5]*delta_j[0+ii*3] + partgrad1[ii*8+5]));
  atomicadd(&(node_f.force[(1+ii*3)*para.number_of_nodes + node_index[5]]), (delta[5]*delta_j[1+ii*3] + partgrad2[ii*8+5]));
  atomicadd(&(node_f.force[(2+ii*3)*para.number_of_nodes + node_index[5]]), (delta[5]*delta_j[2+ii*3] + partgrad3[ii*8+5]));
                                                                                                    
  atomicadd(&(node_f.force[(0+ii*3)*para.number_of_nodes + node_index[6]]), (delta[6]*delta_j[0+ii*3] + partgrad1[ii*8+6]));
  atomicadd(&(node_f.force[(1+ii*3)*para.number_of_nodes + node_index[6]]), (delta[6]*delta_j[1+ii*3] + partgrad2[ii*8+6]));
  atomicadd(&(node_f.force[(2+ii*3)*para.number_of_nodes + node_index[6]]), (delta[6]*delta_j[2+ii*3] + partgrad3[ii*8+6]));
                                                                                                    
  atomicadd(&(node_f.force[(0+ii*3)*para.number_of_nodes + node_index[7]]), (delta[7]*delta_j[0+ii*3] + partgrad1[ii*8+7]));
  atomicadd(&(node_f.force[(1+ii*3)*para.number_of_nodes + node_index[7]]), (delta[7]*delta_j[1+ii*3] + partgrad2[ii*8+7]));
  atomicadd(&(node_f.force[(2+ii*3)*para.number_of_nodes + node_index[7]]), (delta[7]*delta_j[2+ii*3] + partgrad3[ii*8+7]));
 }
}
#ifdef SHANCHEN
/*********************************************************/
/** \name System setup and Kernel funktions */
/*********************************************************/
/**kernel to calculate local populations from hydrodynamic fields given by the tcl values.
 * The mapping is given in terms of the equilibrium distribution.
 *
 * Eq. (2.15) Ladd, J. Fluid Mech. 271, 295-309 (1994)
 * Eq. (4) in Berk Usta, Ladd and Butler, JCP 122, 094902 (2005)
 *
 * @param n_a		 Pointer to the lattice site (Input).
*/
__global__ void calc_n_equilibrium(LB_nodes_gpu n_a) {

  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x;
  if(index<para.number_of_nodes){
  #pragma unroll
  for(int ii=0;ii<SHANCHEN;++ii) { 


    /** default values for fields in lattice units */

    float Rho = para.rho[ii]*para.agrid*para.agrid*para.agrid;
    float v[3] = { 0.0f, 0.0f, 0.0f };
    float pi[6] = { Rho*c_sound_sq, 0.0f, Rho*c_sound_sq, 0.0f, 0.0f, Rho*c_sound_sq };

    float rhoc_sq = Rho*c_sound_sq;
    float avg_rho = para.rho[ii]*para.agrid*para.agrid*para.agrid;
    float local_rho, local_j[3], *local_pi, trace;

    local_rho  = Rho;

    local_j[0] = Rho * v[0];
    local_j[1] = Rho * v[1];
    local_j[2] = Rho * v[2];

    local_pi = pi;

    /** reduce the pressure tensor to the part needed here */
    local_pi[0] -= rhoc_sq;
    local_pi[2] -= rhoc_sq;
    local_pi[5] -= rhoc_sq;

    trace = local_pi[0] + local_pi[2] + local_pi[5];

    float rho_times_coeff;
    float tmp1,tmp2;

    /** update the q=0 sublattice */
    n_a.vd[(0 + ii*LBQ ) * para.number_of_nodes + index] = 1.f/3.f * (local_rho-avg_rho) - 1.f/2.f*trace;

    /** update the q=1 sublattice */
    rho_times_coeff = 1.f/18.f * (local_rho-avg_rho);

    n_a.vd[(1 + ii*LBQ ) * para.number_of_nodes + index] = rho_times_coeff + 1.f/6.f*local_j[0] + 1.f/4.f*local_pi[0] - 1.f/12.f*trace;
    n_a.vd[(2 + ii*LBQ ) * para.number_of_nodes + index] = rho_times_coeff - 1.f/6.f*local_j[0] + 1.f/4.f*local_pi[0] - 1.f/12.f*trace;
    n_a.vd[(3 + ii*LBQ ) * para.number_of_nodes + index] = rho_times_coeff + 1.f/6.f*local_j[1] + 1.f/4.f*local_pi[2] - 1.f/12.f*trace;
    n_a.vd[(4 + ii*LBQ ) * para.number_of_nodes + index] = rho_times_coeff - 1.f/6.f*local_j[1] + 1.f/4.f*local_pi[2] - 1.f/12.f*trace;
    n_a.vd[(5 + ii*LBQ ) * para.number_of_nodes + index] = rho_times_coeff + 1.f/6.f*local_j[2] + 1.f/4.f*local_pi[5] - 1.f/12.f*trace;
    n_a.vd[(6 + ii*LBQ ) * para.number_of_nodes + index] = rho_times_coeff - 1.f/6.f*local_j[2] + 1.f/4.f*local_pi[5] - 1.f/12.f*trace;

    /** update the q=2 sublattice */
    rho_times_coeff = 1.f/36.f * (local_rho-avg_rho);

    tmp1 = local_pi[0] + local_pi[2];
    tmp2 = 2.0f*local_pi[1];
    n_a.vd[(7 + ii*LBQ ) * para.number_of_nodes + index]  = rho_times_coeff + 1.f/12.f*(local_j[0]+local_j[1]) + 1.f/8.f*(tmp1+tmp2) - 1.f/24.f*trace;
    n_a.vd[(8 + ii*LBQ ) * para.number_of_nodes + index]  = rho_times_coeff - 1.f/12.f*(local_j[0]+local_j[1]) + 1.f/8.f*(tmp1+tmp2) - 1.f/24.f*trace;
    n_a.vd[(9 + ii*LBQ ) * para.number_of_nodes + index]  = rho_times_coeff + 1.f/12.f*(local_j[0]-local_j[1]) + 1.f/8.f*(tmp1-tmp2) - 1.f/24.f*trace;
    n_a.vd[(10 + ii*LBQ ) * para.number_of_nodes + index] = rho_times_coeff - 1.f/12.f*(local_j[0]-local_j[1]) + 1.f/8.f*(tmp1-tmp2) - 1.f/24.f*trace;

    tmp1 = local_pi[0] + local_pi[5];
    tmp2 = 2.0f*local_pi[3];

    n_a.vd[(11 + ii*LBQ ) * para.number_of_nodes + index] = rho_times_coeff + 1.f/12.f*(local_j[0]+local_j[2]) + 1.f/8.f*(tmp1+tmp2) - 1.f/24.f*trace;
    n_a.vd[(12 + ii*LBQ ) * para.number_of_nodes + index] = rho_times_coeff - 1.f/12.f*(local_j[0]+local_j[2]) + 1.f/8.f*(tmp1+tmp2) - 1.f/24.f*trace;
    n_a.vd[(13 + ii*LBQ ) * para.number_of_nodes + index] = rho_times_coeff + 1.f/12.f*(local_j[0]-local_j[2]) + 1.f/8.f*(tmp1-tmp2) - 1.f/24.f*trace;
    n_a.vd[(14 + ii*LBQ ) * para.number_of_nodes + index] = rho_times_coeff - 1.f/12.f*(local_j[0]-local_j[2]) + 1.f/8.f*(tmp1-tmp2) - 1.f/24.f*trace;

    tmp1 = local_pi[2] + local_pi[5];
    tmp2 = 2.0f*local_pi[4];

    n_a.vd[(15 + ii*LBQ ) * para.number_of_nodes + index] = rho_times_coeff + 1.f/12.f*(local_j[1]+local_j[2]) + 1.f/8.f*(tmp1+tmp2) - 1.f/24.f*trace;
    n_a.vd[(16 + ii*LBQ ) * para.number_of_nodes + index] = rho_times_coeff - 1.f/12.f*(local_j[1]+local_j[2]) + 1.f/8.f*(tmp1+tmp2) - 1.f/24.f*trace;
    n_a.vd[(17 + ii*LBQ ) * para.number_of_nodes + index] = rho_times_coeff + 1.f/12.f*(local_j[1]-local_j[2]) + 1.f/8.f*(tmp1-tmp2) - 1.f/24.f*trace;
    n_a.vd[(18 + ii*LBQ ) * para.number_of_nodes + index] = rho_times_coeff - 1.f/12.f*(local_j[1]-local_j[2]) + 1.f/8.f*(tmp1-tmp2) - 1.f/24.f*trace;

    /**set different seed for randomgen on every node */
    n_a.seed[index] = para.your_seed + index;
  }
  }
}
#endif
/** kernel to set the local density
 *
 * @param n_a		   the current nodes array (double buffering!)
 * @param single_nodeindex the node to set the velocity for
 * @param rho              the density to set
 */
__global__ void set_rho(LB_nodes_gpu n_a,  LB_values_gpu *d_v, int single_nodeindex,float *rho) {

  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x;
  /*Note: this sets the velocities to zero */
  if(index == 0){
     float local_rho;
     #pragma unroll
     for(int ii=0;ii<SHANCHEN;++ii) { 
       /** default values for fields in lattice units */
       local_rho = (rho[ii]-para.rho[ii])*para.agrid*para.agrid*para.agrid;
       d_v[single_nodeindex].rho[ii]=rho[ii];
       n_a.vd[(0  + ii*LBQ ) * para.number_of_nodes + single_nodeindex] = 1.f/3.f * local_rho ;
       n_a.vd[(1  + ii*LBQ ) * para.number_of_nodes + single_nodeindex] = 1.f/18.f * local_rho ;
       n_a.vd[(2  + ii*LBQ ) * para.number_of_nodes + single_nodeindex] = 1.f/18.f * local_rho ;
       n_a.vd[(3  + ii*LBQ ) * para.number_of_nodes + single_nodeindex] = 1.f/18.f * local_rho ;
       n_a.vd[(4  + ii*LBQ ) * para.number_of_nodes + single_nodeindex] = 1.f/18.f * local_rho ;
       n_a.vd[(5  + ii*LBQ ) * para.number_of_nodes + single_nodeindex] = 1.f/18.f * local_rho ;
       n_a.vd[(6  + ii*LBQ ) * para.number_of_nodes + single_nodeindex] = 1.f/18.f * local_rho ;
       n_a.vd[(7  + ii*LBQ ) * para.number_of_nodes + single_nodeindex] = 1.f/36.f * local_rho ;
       n_a.vd[(8  + ii*LBQ ) * para.number_of_nodes + single_nodeindex] = 1.f/36.f * local_rho ;
       n_a.vd[(9  + ii*LBQ ) * para.number_of_nodes + single_nodeindex] = 1.f/36.f * local_rho ;
       n_a.vd[(10 + ii*LBQ ) * para.number_of_nodes + single_nodeindex] = 1.f/36.f * local_rho ;
       n_a.vd[(11 + ii*LBQ ) * para.number_of_nodes + single_nodeindex] = 1.f/36.f * local_rho ;
       n_a.vd[(12 + ii*LBQ ) * para.number_of_nodes + single_nodeindex] = 1.f/36.f * local_rho ;
       n_a.vd[(13 + ii*LBQ ) * para.number_of_nodes + single_nodeindex] = 1.f/36.f * local_rho ;
       n_a.vd[(14 + ii*LBQ ) * para.number_of_nodes + single_nodeindex] = 1.f/36.f * local_rho ;
       n_a.vd[(15 + ii*LBQ ) * para.number_of_nodes + single_nodeindex] = 1.f/36.f * local_rho ;
       n_a.vd[(16 + ii*LBQ ) * para.number_of_nodes + single_nodeindex] = 1.f/36.f * local_rho ;
       n_a.vd[(17 + ii*LBQ ) * para.number_of_nodes + single_nodeindex] = 1.f/36.f * local_rho ;
       n_a.vd[(18 + ii*LBQ ) * para.number_of_nodes + single_nodeindex] = 1.f/36.f * local_rho ;
     }
  }
}


/** kernel to calculate local populations from hydrodynamic fields
 * from given flow field velocities.  The mapping is given in terms of
 * the equilibrium distribution.
 *
 * Eq. (2.15) Ladd, J. Fluid Mech. 271, 295-309 (1994)
 * Eq. (4) in Berk Usta, Ladd and Butler, JCP 122, 094902 (2005)
 *
 * @param n_a		   the current nodes array (double buffering!)
 * @param single_nodeindex the node to set the velocity for
 * @param velocity         the velocity to set
 */
__global__ void set_u_equilibrium(LB_nodes_gpu n_a, int single_nodeindex,float *velocity) {

  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x;

  if(index == 0){
  float v[3];
  float mode[4*SHANCHEN];
  float rhoc_sq,avg_rho;
  float local_rho, local_j[3], *local_pi, trace;
  v[0] = velocity[0];
  v[1] = velocity[1];
  v[2] = velocity[2];
  #pragma unroll
  for(int ii=0;ii<SHANCHEN;++ii) { 

    /** default values for fields in lattice units */
    calc_mode(&mode[4*ii], n_a, single_nodeindex,ii);
    float Rho = mode[0*4*ii] + para.rho[ii]*para.agrid*para.agrid*para.agrid; 

    float pi[6] = { Rho*c_sound_sq, 0.0f, Rho*c_sound_sq, 0.0f, 0.0f, Rho*c_sound_sq };

    rhoc_sq = Rho*c_sound_sq;
    avg_rho = para.rho[ii]*para.agrid*para.agrid*para.agrid;

    local_rho  = Rho;

    local_j[0] = Rho * v[0];
    local_j[1] = Rho * v[1];
    local_j[2] = Rho * v[2];

    local_pi = pi;

    /** reduce the pressure tensor to the part needed here */
    local_pi[0] -= rhoc_sq; 
    local_pi[2] -= rhoc_sq;
    local_pi[5] -= rhoc_sq;

    trace = local_pi[0] + local_pi[2] + local_pi[5];

    float rho_times_coeff;
    float tmp1,tmp2;

    /** update the q=0 sublattice */
    n_a.vd[(0 + ii*LBQ ) * para.number_of_nodes + single_nodeindex] = 1.f/3.f * (local_rho-avg_rho) - 1.f/2.f*trace;

    /** update the q=1 sublattice */
    rho_times_coeff = 1.f/18.f * (local_rho-avg_rho);

    n_a.vd[(1 + ii*LBQ ) * para.number_of_nodes + single_nodeindex] = rho_times_coeff + 1.f/6.f*local_j[0] + 1.f/4.f*local_pi[0] - 1.f/12.f*trace;
    n_a.vd[(2 + ii*LBQ ) * para.number_of_nodes + single_nodeindex] = rho_times_coeff - 1.f/6.f*local_j[0] + 1.f/4.f*local_pi[0] - 1.f/12.f*trace;
    n_a.vd[(3 + ii*LBQ ) * para.number_of_nodes + single_nodeindex] = rho_times_coeff + 1.f/6.f*local_j[1] + 1.f/4.f*local_pi[2] - 1.f/12.f*trace;
    n_a.vd[(4 + ii*LBQ ) * para.number_of_nodes + single_nodeindex] = rho_times_coeff - 1.f/6.f*local_j[1] + 1.f/4.f*local_pi[2] - 1.f/12.f*trace;
    n_a.vd[(5 + ii*LBQ ) * para.number_of_nodes + single_nodeindex] = rho_times_coeff + 1.f/6.f*local_j[2] + 1.f/4.f*local_pi[5] - 1.f/12.f*trace;
    n_a.vd[(6 + ii*LBQ ) * para.number_of_nodes + single_nodeindex] = rho_times_coeff - 1.f/6.f*local_j[2] + 1.f/4.f*local_pi[5] - 1.f/12.f*trace;

    /** update the q=2 sublattice */
    rho_times_coeff = 1.f/36.f * (local_rho-avg_rho);

    tmp1 = local_pi[0] + local_pi[2];
    tmp2 = 2.0f*local_pi[1];
    n_a.vd[(7 + ii*LBQ ) * para.number_of_nodes + single_nodeindex]  = rho_times_coeff + 1.f/12.f*(local_j[0]+local_j[1]) + 1.f/8.f*(tmp1+tmp2) - 1.f/24.f*trace;
    n_a.vd[(8 + ii*LBQ ) * para.number_of_nodes + single_nodeindex]  = rho_times_coeff - 1.f/12.f*(local_j[0]+local_j[1]) + 1.f/8.f*(tmp1+tmp2) - 1.f/24.f*trace;
    n_a.vd[(9 + ii*LBQ ) * para.number_of_nodes + single_nodeindex]  = rho_times_coeff + 1.f/12.f*(local_j[0]-local_j[1]) + 1.f/8.f*(tmp1-tmp2) - 1.f/24.f*trace;
    n_a.vd[(10 + ii*LBQ ) * para.number_of_nodes + single_nodeindex] = rho_times_coeff - 1.f/12.f*(local_j[0]-local_j[1]) + 1.f/8.f*(tmp1-tmp2) - 1.f/24.f*trace;

    tmp1 = local_pi[0] + local_pi[5];
    tmp2 = 2.0f*local_pi[3];

    n_a.vd[(11 + ii*LBQ ) * para.number_of_nodes + single_nodeindex] = rho_times_coeff + 1.f/12.f*(local_j[0]+local_j[2]) + 1.f/8.f*(tmp1+tmp2) - 1.f/24.f*trace;
    n_a.vd[(12 + ii*LBQ ) * para.number_of_nodes + single_nodeindex] = rho_times_coeff - 1.f/12.f*(local_j[0]+local_j[2]) + 1.f/8.f*(tmp1+tmp2) - 1.f/24.f*trace;
    n_a.vd[(13 + ii*LBQ ) * para.number_of_nodes + single_nodeindex] = rho_times_coeff + 1.f/12.f*(local_j[0]-local_j[2]) + 1.f/8.f*(tmp1-tmp2) - 1.f/24.f*trace;
    n_a.vd[(14 + ii*LBQ ) * para.number_of_nodes + single_nodeindex] = rho_times_coeff - 1.f/12.f*(local_j[0]-local_j[2]) + 1.f/8.f*(tmp1-tmp2) - 1.f/24.f*trace;

    tmp1 = local_pi[2] + local_pi[5];
    tmp2 = 2.0f*local_pi[4];

    n_a.vd[(15 + ii*LBQ ) * para.number_of_nodes + single_nodeindex] = rho_times_coeff + 1.f/12.f*(local_j[1]+local_j[2]) + 1.f/8.f*(tmp1+tmp2) - 1.f/24.f*trace;
    n_a.vd[(16 + ii*LBQ ) * para.number_of_nodes + single_nodeindex] = rho_times_coeff - 1.f/12.f*(local_j[1]+local_j[2]) + 1.f/8.f*(tmp1+tmp2) - 1.f/24.f*trace;
    n_a.vd[(17 + ii*LBQ ) * para.number_of_nodes + single_nodeindex] = rho_times_coeff + 1.f/12.f*(local_j[1]-local_j[2]) + 1.f/8.f*(tmp1-tmp2) - 1.f/24.f*trace;
    n_a.vd[(18 + ii*LBQ ) * para.number_of_nodes + single_nodeindex] = rho_times_coeff - 1.f/12.f*(local_j[1]-local_j[2]) + 1.f/8.f*(tmp1-tmp2) - 1.f/24.f*trace;

  }
  }
}
/**calculate mass of the whole fluid kernel
 * @param *sum				Pointer to result storage value (Output)
 * @param n_a				Pointer to local node residing in array a (Input)
*/
__global__ void calc_mass(LB_nodes_gpu n_a, float *sum) {
  float mode[4];

  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x;

  #pragma unroll
  for(int ii=0;ii<SHANCHEN;++ii) { 
  if(index<para.number_of_nodes){
    calc_mode(mode, n_a, index,ii);
    float Rho = mode[0] + para.rho[ii]*para.agrid*para.agrid*para.agrid;
    atomicadd(&(sum[0]), Rho);
  }
}
}

/** (re-)initialization of the node force / set up of external force in lb units
 * @param node_f		Pointer to local node force (Input)
*/
__global__ void reinit_node_force(LB_node_force_gpu node_f){

  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x;

  if(index<para.number_of_nodes){
   #pragma unroll
   for(int ii=0;ii<SHANCHEN;++ii){
#ifdef EXTERNAL_FORCE
    if(para.external_force){
      node_f.force[(0+ii*3)*para.number_of_nodes + index] = para.ext_force[0]*powf(para.agrid,4)*para.tau*para.tau;
      node_f.force[(1+ii*3)*para.number_of_nodes + index] = para.ext_force[1]*powf(para.agrid,4)*para.tau*para.tau;
      node_f.force[(2+ii*3)*para.number_of_nodes + index] = para.ext_force[2]*powf(para.agrid,4)*para.tau*para.tau;
    }
    else{
      node_f.force[(0+ii*3)*para.number_of_nodes + index] = 0.0f;
      node_f.force[(1+ii*3)*para.number_of_nodes + index] = 0.0f;
      node_f.force[(2+ii*3)*para.number_of_nodes + index] = 0.0f;
    }
#else
    node_f.force[(0+ii*3)*para.number_of_nodes + index] = 0.0f;
    node_f.force[(1+ii*3)*para.number_of_nodes + index] = 0.0f;
    node_f.force[(2+ii*3)*para.number_of_nodes + index] = 0.0f;
#endif
   }
  }
}


/**set extern force on single nodes kernel
 * @param n_extern_nodeforces		number of nodes (Input)
 * @param *extern_nodeforces		Pointer to extern node force array (Input)
 * @param node_f			node force struct (Output)
*/
__global__ void init_extern_nodeforces(int n_extern_nodeforces, LB_extern_nodeforce_gpu *extern_nodeforces, LB_node_force_gpu node_f){

  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x;

  if(index<n_extern_nodeforces){
   #pragma unroll
   for(int ii=0;ii<SHANCHEN;++ii){
    node_f.force[(0+ii*3)*para.number_of_nodes + extern_nodeforces[index].index] = extern_nodeforces[index].force[0]*powf(para.agrid,4)*para.tau*para.tau;
    node_f.force[(1+ii*3)*para.number_of_nodes + extern_nodeforces[index].index] = extern_nodeforces[index].force[1]*powf(para.agrid,4)*para.tau*para.tau;
    node_f.force[(2+ii*3)*para.number_of_nodes + extern_nodeforces[index].index] = extern_nodeforces[index].force[2]*powf(para.agrid,4)*para.tau*para.tau;
   }
  }
}

#endif //SHANCHEN
/** kernel for the initalisation of the particle force array
 * @param *particle_force	Pointer to local particle force (Output)
 * @param *part			Pointer to the particle rn seed storearray (Output)
*/
__global__ void init_particle_force(LB_particle_force_gpu *particle_force, LB_particle_seed_gpu *part){
	
  unsigned int part_index = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x;
	
  if(part_index<devpara.number_of_particles){
    particle_force[part_index].f[0] = 0.0f;
    particle_force[part_index].f[1] = 0.0f;
    particle_force[part_index].f[2] = 0.0f;
	
    part[part_index].seed = para.your_seed + part_index;
  }
			
}

/** kernel for the initalisation of the partikel force array
 * @param *particle_force	pointer to local particle force (Input)
*/
__global__ void reset_particle_force(LB_particle_force_gpu *particle_force){
	
  unsigned int part_index = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x;
	
  if(part_index<devpara.number_of_particles){
    particle_force[part_index].f[0] = 0.0f;
    particle_force[part_index].f[1] = 0.0f;
    particle_force[part_index].f[2] = 0.0f;
  }			
}

/**set the boundary flag for all boundary nodes
 * @param *boundindex	     	Pointer to the 1d index of the boundnode (Input)
 * @param number_of_boundnodes	The number of boundary nodes
 * @param n_a			Pointer to local node residing in array a (Input)
 * @param n_b			Pointer to local node residing in array b (Input)
*/
__global__ void init_boundaries(int *boundindex, int number_of_boundnodes, LB_nodes_gpu n_a, LB_nodes_gpu n_b){

  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x;

  if(index<number_of_boundnodes){
    n_a.boundary[boundindex[index]] = n_b.boundary[boundindex[index]] = 1;
  }	
}


/**set extern force on single nodes kernel
 * @param n_extern_nodeforces		number of nodes (Input)
 * @param *extern_nodeforces		Pointer to extern node force array (Input)
 * @param node_f			node force struct (Output)
*/
__global__ void init_boundaries(int *boundary_node_list, int *boundary_index_list, int number_of_boundnodes, LB_nodes_gpu n_a, LB_nodes_gpu n_b){

  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x;

  if(index<number_of_boundnodes){
    n_a.boundary[boundary_node_list[index]] = boundary_index_list[index]+1;
    n_b.boundary[boundary_node_list[index]] = boundary_index_list[index]+1;
  }
}

/**reset the boundary flag of every node
 * @param n_a		Pointer to local node residing in array a (Input)
 * @param n_b		Pointer to local node residing in array b (Input)	
*/
__global__ void reset_boundaries(LB_nodes_gpu n_a, LB_nodes_gpu n_b){

  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x;

  if(index<para.number_of_nodes){
    n_a.boundary[index] = n_b.boundary[index] = 0;
  }
}

/** integrationstep of the lb-fluid-solver
 * @param n_a		Pointer to local node residing in array a (Input)
 * @param n_b		Pointer to local node residing in array b (Input)
 * @param *d_v		Pointer to local device values (Input)
 * @param node_f	Pointer to local node force (Input)
*/
__global__ void integrate(LB_nodes_gpu n_a, LB_nodes_gpu n_b, LB_values_gpu *d_v, LB_node_force_gpu node_f, float* buffer, int *d_gpu_n){
  /**every node is connected to a thread via the index*/
  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x;
  /**the 19 moments (modes) are only temporary register values */
//#ifndef SHANCHEN
  float mode[19];
//#else //SHANCHEN
//  float mode[19*SHANCHEN];
//#endif
  LB_randomnr_gpu rng;
//printf("#nodes %i\n", para.number_of_nodes);
  if(index<para.number_of_nodes){
    /** storing the seed into a register value*/
    rng.seed = n_a.seed[index];
    /**calc_m_from_n*/
    calc_m_from_n(n_a, index, mode);
    /**lb_relax_modes*/
#ifndef SHANCHEN 
    //relax_modes(mode, index, node_f);
#else
    //relax_modes(mode, index, node_f, d_v);
#endif
#if 0 
    /**lb_thermalize_modes */
    if (para.fluct) thermalize_modes(mode, index, &rng);
#ifdef EXTERNAL_FORCES
    /**if external force is used apply node force */
#ifndef SHANCHEN 
    apply_forces(index, mode, node_f);
#else // SHANCHEN
    apply_forces(index, mode, node_f, d_v);
#endif //SHANCHEN
#else
    /**if partcles are used apply node forces*/
#ifndef SHANCHEN 
    if (devpara.number_of_particles) apply_forces(index, mode, node_f); 
#else // SHANCHEN
    if (devpara.number_of_particles) apply_forces(index, mode, node_f, d_v); 
#endif //SHANCHEN
#endif
#endif // 0
    /**lb_calc_n_from_modes_push*/
    normalize_modes(mode);
    /**calc of velocity densities and streaming with pbc*/
    //if (*d_gpu_n ==1)
      calc_n_from_modes_push(n_b, mode, index);
    //else{
      if (devpara.number_of_gpus > 1) calc_n_from_modes_buffer(n_b, buffer, mode, index);
    /** rewriting the seed back to the global memory*/
    n_b.seed[index] = rng.seed;
  }  
}

/** integrationstep of the lb-fluid-solver
 * @param n_a		Pointer to local node residing in array a (Input)
 * @param n_b		Pointer to local node residing in array b (Input)
 * @param *d_v		Pointer to local device values (Input)
 * @param node_f	Pointer to local node force (Input)
*/
__global__ void write_buffer(LB_nodes_gpu n_c, float* buffer){

  /**every node is connected to a thread via the index*/
  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x;
  /**the 19 moments (modes) are only temporary register values */

  if(index<para.number_of_nodes){
    //store buffer values at thier destination in lb nodes struct
    write_n_from_buffer(n_c, buffer, index);
  }
}
__global__ void init_buf(float* s_buf, float* r_buf){

  /**every node is connected to a thread via the index*/
  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x;
//if(index == 0)printf("imax %i\n", (5*2*(para.number_of_halo_nodes[0]+para.number_of_halo_nodes[1]+para.number_of_halo_nodes[2])));
  if(index<(5*2*(para.number_of_halo_nodes[0]+para.number_of_halo_nodes[1]+para.number_of_halo_nodes[2]))){
    //store buffer values at thier destination in lb nodes struct
    s_buf[index] = 0.0;
    r_buf[index] = 0.0;
  }
}
/** part interaction kernel
 * @param n_a				Pointer to local node residing in array a (Input)
 * @param *particle_data		Pointer to the particle position and velocity (Input)
 * @param *particle_force		Pointer to the particle force (Input)
 * @param *part				Pointer to the rn array of the particles (Input)
 * @param node_f			Pointer to local node force (Input)
*/
#ifndef SHANCHEN
__global__ void calc_fluid_particle_ia(LB_nodes_gpu n_a, LB_particle_gpu *particle_data, LB_particle_force_gpu *particle_force, LB_node_force_gpu node_f, LB_particle_seed_gpu *part){
#else 
__global__ void calc_fluid_particle_ia(LB_nodes_gpu n_a, LB_particle_gpu *particle_data, LB_particle_force_gpu *particle_force, LB_node_force_gpu node_f, LB_particle_seed_gpu *part,LB_values_gpu *d_v){
#endif
	
  unsigned int part_index = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int node_index[8];
  float delta[8];
#ifndef SHANCHEN
  float delta_j[3]; 
#else // SHANCHEN
  float delta_j[3*SHANCHEN]; 
  float partgrad1[8*SHANCHEN]; 
  float partgrad2[8*SHANCHEN]; 
  float partgrad3[8*SHANCHEN]; 
#endif //SHANCHEN
  LB_randomnr_gpu rng_part;
  if(part_index<devpara.number_of_particles){

    rng_part.seed = part[part_index].seed;
    /**force acting on the particle. delta_j will be used later to compute the force that acts back onto the fluid. */
#ifndef SHANCHEN
    calc_viscous_force(n_a, delta, particle_data, particle_force, part_index, &rng_part, delta_j, node_index);
    calc_node_force(delta, delta_j, node_index, node_f); 
#else 
    calc_viscous_force(n_a, delta, partgrad1, partgrad2, partgrad3, particle_data, particle_force, part_index, &rng_part, delta_j, node_index,d_v);
    calc_node_force(delta, delta_j, partgrad1, partgrad2, partgrad3, node_index, node_f); 
#endif 
    /**force which acts back to the fluid node */
    part[part_index].seed = rng_part.seed;		
  }
}
/**Bounce back boundary read kernel
  * @param n_a Pointer to local node residing in array a (Input)
  * @param n_b Pointer to local node residing in array b (Input)
  */
__global__ void bb_read(LB_nodes_gpu n_a, LB_nodes_gpu n_b, float* lb_boundary_velocity, float* lb_boundary_force){

    unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x;

      if(index<para.number_of_nodes){
#ifdef SHANCHEN
            bounce_back_read_shanchen(n_b, n_a, index);
#else
            bounce_back_read(n_b, n_a, index, lb_boundary_velocity, lb_boundary_force);
#endif
      }
}
/**Bounce back boundary write kernel
 * @param n_a					Pointer to local node residing in array a (Input)
 * @param n_b					Pointer to local node residing in array b (Input)
*/
__global__ void bb_write(LB_nodes_gpu n_a, LB_nodes_gpu n_b){

  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x;

  if(index<para.number_of_nodes){
    bounce_back_write(n_b, n_a, index);
  }
}

/** get physical values of the nodes (density, velocity, ...)
 * @param n_a		Pointer to local node residing in array a (Input)
 * @param *d_v		Pointer to local device values (Input)
*/
#ifndef SHANCHEN
__global__ void values(LB_nodes_gpu n_a, LB_values_gpu *d_v, LB_node_force_gpu node_f){

  unsigned int singlenode = 0;
  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x;

  if(index<para.number_of_nodes){
    float mode[4];
    calc_mode(mode, n_a, index);
    calc_values(n_a, mode, d_v, index, singlenode, node_f);
  }
}
/** get physical values of the nodes without halonodes (density, velocity, ...)
 * @param n_a		Pointer to local node residing in array a (Input)
 * @param *d_v		Pointer to local device values (Input)
*/
__global__ void values_wo_halo(LB_nodes_gpu n_a, LB_values_gpu *d_v, LB_node_force_gpu node_f){

  unsigned int singlenode = 0;
  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x;

  if(index<para.number_of_nodes){
    float mode[4];
    calc_mode(mode, n_a, index);
    calc_values_wo_halo(n_a, mode, d_v, index, singlenode, node_f);
  }
}
#else // SHANCHEN
__global__ void values(LB_nodes_gpu n_a, LB_values_gpu *d_v, LB_node_force_gpu node_f){
 /* NOTE: in SHANCHEN d_v are updated in relax_modes() because the
 forces are needed, which are reset to zero (or to the ext. force
 value) in apply_forces(), at the end of the LB loop. When a request
 to print values comes from tcl, one just needs to copy the data,
 without having to recompute forces and the field. This is why this
 kernel is in general not called in the SHANCHEN implementation.
 The only execption is at the initialization, lb_init_GPU()  FIXME: The idea behind
 is changed quite a bit wrt the old LB, one should think if another
 approach is better (i.e. unifying the LB and SC way of calculationg values)
*/
  unsigned int singlenode = 0;
  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x;

  if(index<para.number_of_nodes){
    float mode[4*SHANCHEN];
    #pragma unroll
    for(int ii=0;ii<SHANCHEN;++ii){
      calc_mode(&mode[0+4*ii], n_a, index,ii);
    }
    calc_values(n_a, mode, d_v, node_f, index, singlenode);
  }
}
#endif // SHANCHEN

/** get boundary flags
 *  @param n_a	              Pointer to local node residing in array a (Input)
 *  @param device_bound_array Pointer to local device values (Input)
 */
__global__ void lb_get_boundaries(LB_nodes_gpu n_a, unsigned int *device_bound_array){

  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x;

  if(index<para.number_of_nodes){
   device_bound_array[index] = n_a.boundary[index];
  }
}

/** get boundary flags
 *  @param n_a	              Pointer to local node residing in array a (Input)
 *  @param device_bound_array Pointer to local device values (Input)
 */
__global__ void lb_get_boundaries_wo_halo(LB_nodes_gpu n_a, unsigned int *device_bound_array){

  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x;

  if(index<para.number_of_nodes){

    unsigned int iindex;
    unsigned int xyz[3];
    index_to_xyz(index, xyz);
    unsigned int x = xyz[0];
    unsigned int y = xyz[1];
    unsigned int z = xyz[2];
    iindex=index;
    if(x != 0 && x != (para.dim_x-1) && y != 0 && y != (para.dim_y-1) && z != 0 && z != (para.dim_z-1)){
     iindex = (x-1) + (y-1)*(para.dim_x-2) + (z-1)*(para.dim_x-2)*(para.dim_y-2);
     device_bound_array[iindex] = n_a.boundary[index];
     //printf("bounds %i index %i\n", device_bound_array[iindex], iindex);
    }
  }
}

/**print single node values kernel
 * @param single_nodeindex		index of the node (Input)
 * @param *d_p_v			Pointer to result storage array (Input)
 * @param n_a				Pointer to local node residing in array a (Input)
*/
#ifndef SHANCHEN
__global__ void lb_print_node(int single_nodeindex, LB_values_gpu *d_p_v, LB_nodes_gpu n_a, LB_node_force_gpu node_f){

  float mode[19];
  unsigned int singlenode = 1;
  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x;
  if(index == 0){
    calc_m_from_n(n_a, single_nodeindex, mode);
    calc_values(n_a, mode, d_p_v, single_nodeindex, singlenode, node_f);
  }
}
#else //SHANCHEN
__global__ void lb_print_node(int single_nodeindex, LB_values_gpu *d_p_v, LB_nodes_gpu n_a, LB_values_gpu * d_v){
	
  float mode[4*SHANCHEN];
  unsigned int singlenode = 1;
  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x;
  if(index == 0){ // in SHANCHEN values are computed in relax_modes() so here we need just to copy the value.
    #pragma unroll
    for(int ii=0;ii<SHANCHEN;++ii){
       d_p_v[0].rho[ii] =  d_v[single_nodeindex].rho[ii];
    }
    d_p_v[0].v[0] = d_v[single_nodeindex].v[0] ;
    d_p_v[0].v[1] = d_v[single_nodeindex].v[1] ;
    d_p_v[0].v[2] = d_v[single_nodeindex].v[2] ;
  }
}
#endif
/**calculate momentum of the hole fluid kernel
 * @param node_f			node force struct (Input)
 * @param *sum				Pointer to result storage value (Output)
 * @param n_a				Pointer to local node residing in array a (Input)
*/
#ifndef SHANCHEN
__global__ void momentum(LB_nodes_gpu n_a, float *sum, LB_node_force_gpu node_f) {
  float mode[4];

  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x;

  if(index<para.number_of_nodes){
    calc_mode(mode, n_a, index);
    if(n_a.boundary[index]){
      mode[1] = mode[2] = mode[3] = 0.f;
    }
    atomicadd(&(sum[0]), mode[1]+node_f.force[0*para.number_of_nodes + index]);
    atomicadd(&(sum[1]), mode[2]+node_f.force[1*para.number_of_nodes + index]);
    atomicadd(&(sum[2]), mode[3]+node_f.force[2*para.number_of_nodes + index]);
  }
}
#else // SHANCHEN

__global__ void momentum(LB_nodes_gpu n_a, float *sum, LB_node_force_gpu node_f) {
  float mode[4];

  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x;

  if(index<para.number_of_nodes){
   #pragma unroll
   for(int ii=0;ii<SHANCHEN;++ii){
    calc_mode(mode, n_a, index,ii);
    if(n_a.boundary[index]){
      mode[1] = mode[2] = mode[3] = 0.f;
    }
    atomicadd(&(sum[0]), mode[1]+node_f.force[(0+ii*3)*para.number_of_nodes + index]); 
    atomicadd(&(sum[1]), mode[2]+node_f.force[(1+ii*3)*para.number_of_nodes + index]);
    atomicadd(&(sum[2]), mode[3]+node_f.force[(2+ii*3)*para.number_of_nodes + index]);
   }
  }
}

#endif

/**print single node boundary flag
 * @param single_nodeindex		index of the node (Input)
 * @param *device_flag			Pointer to result storage array (Input)
 * @param n_a				Pointer to local node residing in array a (Input)
*/
__global__ void lb_get_boundary_flag(int single_nodeindex, unsigned int *device_flag, LB_nodes_gpu n_a){

  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x;

  if(index == 0){
    device_flag[0] = n_a.boundary[single_nodeindex];
  }
}
/*********************************************************/
/** \name Host functions to setup and call kernels */
/*********************************************************/
/**********************************************************************/
/* Host funktions to setup and call kernels*/
/**********************************************************************/

/**get hardware info of GPUs
 * @param dev device number
*/
void hw::check_dev(int dev){
  
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, dev);
   
  printf("Major revision number:         %d\n",  prop.major);
  printf("Minor revision number:         %d\n",  prop.minor);
  printf("Name:                          %s\n",  prop.name);
  printf("Total global memory:           %u\n",  prop.totalGlobalMem);
  printf("Total shared memory per block: %u\n",  prop.sharedMemPerBlock);
  printf("Total registers per block:     %d\n",  prop.regsPerBlock);
  printf("Warp size:                     %d\n",  prop.warpSize);
  printf("Maximum memory pitch:          %u\n",  prop.memPitch);
  printf("Maximum threads per block:     %d\n",  prop.maxThreadsPerBlock);
  for (int i = 0; i < 3; ++i)
     printf("Maximum dimension %d of block:  %d\n", i, prop.maxThreadsDim[i]);
  for (int i = 0; i < 3; ++i)
     printf("Maximum dimension %d of grid:   %d\n", i, prop.maxGridSize[i]);
  printf("Clock rate:                    %d\n",  prop.clockRate);
  printf("Total constant memory:         %u\n",  prop.totalConstMem);
  printf("Texture alignment:             %u\n",  prop.textureAlignment);
  printf("Concurrent copy and execution: %s\n",  (prop.deviceOverlap ? "Yes" : "No"));
  printf("Number of multiprocessors:     %d\n",  prop.multiProcessorCount);
  printf("Kernel execution timeout:      %s\n",  (prop.kernelExecTimeoutEnabled ? "Yes" : "No"));
  
  if(prop.unifiedAddressing == 1){
     printf("UVA possible: yes\n");
  }else{
     printf("UVA possible: no\n");
  }
}
/**get hardware info of GPUs
 * @param lbpar_gpu.number_of_gpus
*/
void hw::get_dev_count(){
  
  cuda_check_errors(cudaGetDeviceCount(&lbdevicepar_gpu.number_of_gpus));

}
/**get hardware info of GPUs
 * @param dev device number
*/
void hw::set_dev(int dev){

  cuda_check_errors(cudaSetDevice(dev)); 
  //printf("host no. %i set gpu no. %i \n", this_node, dev);

}
/**get hardware info of GPUs
 * @param dev device number
*/
int lbgpu::set_devices(int* dev, int count){

  lbdevicepar_gpu.number_of_gpus = count;
  //printf("number of GPUs %i \n", count);
  return ES_OK;
}

/**get hardware info of GPUs
 * @param dev device number
*/
int lbgpu::get_devices(int* dev){

  int count;
  count = lbdevicepar_gpu.number_of_gpus;
  //printf("number of GPUs %i \n", count);
  return count;
}

void lbgpu::reinit_plan(){

  LB_TRACE(printf("node %i reinit_plan: number of gpus %i\n", this_node, lbdevicepar_gpu.number_of_gpus));
//only one gpu per cpu node so far!!!
  lbdevicepar_gpu.gpus_per_cpu = 1;
  if(plan_initflag == 1){
    free(plan);
  }
  //lbpar_gpu.number_of_gpus = count;
//check if number of nodes suits to number of gpus
  if(lbdevicepar_gpu.number_of_gpus == 1){
    if(this_node == 0){
      lbdevicepar_gpu.gpu_number = lbdevicepar_gpu.devices[0];
      //malloc plan struct for each gpu per cpu node 
      plan = (plan_gpu*)malloc(lbdevicepar_gpu.gpus_per_cpu*sizeof(plan_gpu));
      gpu_n = lbdevicepar_gpu.gpus_per_cpu;
      //printf("thisnode %i gpun %i\n",this_node, gpu_n);
      for(int g = 0; g < gpu_n; ++g){
        plan[g].initflag = 0;
      }
    }
  }else{
    if (n_nodes%lbdevicepar_gpu.number_of_gpus == 1 || lbdevicepar_gpu.number_of_gpus%n_nodes == 1){
      printf("ERROR: Number of MPI process must be multiple of number of GPUs!!!\n");
      exit(-1);
    }
//   so far one needs at least #n mpi process to use #n gpus
//  distribute gpu to cpu nodes  
    //lbdevicepar_gpu.gpu_number = this_node%lbpar_gpu.number_of_gpus;
    //printf("thisnode %i devs %i %i\n",this_node, lbpar_gpu.devices[0], lbpar_gpu.devices[1]);
    lbdevicepar_gpu.gpu_number = lbdevicepar_gpu.devices[this_node%lbdevicepar_gpu.number_of_gpus];
    printf("thisnode %i gpu_number %i\n",this_node, lbdevicepar_gpu.gpu_number);
    //printf("par gpu dimx address %p \n", &lbpar_gpu.dim_x);
    //printf("GPU number: %i -> this_node %i\n", lbdevicepar_gpu.gpu_number, this_node);
    hw::set_dev(lbdevicepar_gpu.gpu_number);
    //malloc plan struct for each gpu per cpu node 
    plan = (plan_gpu*)malloc(lbdevicepar_gpu.gpus_per_cpu*sizeof(plan_gpu));
    gpu_n = lbdevicepar_gpu.gpus_per_cpu;
    plan_initflag = 1;
    //printf("thisnode %i gpun %i\n",this_node, gpu_n);
    for(int g = 0; g < gpu_n; ++g){
      plan[g].initflag = 0;
    }
  }
}

void lbgpu::setup_plan(){

  LB_TRACE(printf("node %i setup_plan gpu %i\n", this_node, lbdevicepar_gpu.gpu_number));
//only one gpu per cpu node so far!!!
  lbdevicepar_gpu.gpus_per_cpu = 1;
  hw::get_dev_count();
//check if number of nodes suits to number of gpus
  if (n_nodes%lbdevicepar_gpu.number_of_gpus == 1 || lbdevicepar_gpu.number_of_gpus%n_nodes == 1){
    printf("ERROR: Number of MPI process must be multiple of number of GPUs!!!\n");
    exit(-1);
  }
// so far one needs at least #n mpi process to use #n gpus
//distribute gpu to cpu nodes  
  for(int i = 0; i<n_nodes; ++i){
    /* decided which CPU nodes belongs to which GPU*/
    if (this_node == i) {
      lbdevicepar_gpu.gpu_number = this_node%lbdevicepar_gpu.number_of_gpus;
      //printf("par gpu dimx address %p \n", &lbpar_gpu.dim_x);
      //printf("GPU number: %i -> this_node %i\n", lbpar_gpu.gpu_number, this_node);
      hw::set_dev(lbdevicepar_gpu.gpu_number);
    }
  }
  //malloc plan struct for each gpu per cpu node 
  plan = (plan_gpu*)malloc(lbdevicepar_gpu.gpus_per_cpu*sizeof(plan_gpu));
  gpu_n = lbdevicepar_gpu.gpus_per_cpu;
  plan_initflag=1;
  //printf("thisnode %i gpun %i\n",this_node, gpu_n);
  for(int g = 0; g < gpu_n; ++g){
    plan[g].initflag = 0;
  }
}

__global__ void init_arr_1(float *s_buf_d, int al){

  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x;
  if(index < al){
    s_buf_d[index] = 1.0;
  }

}
__global__ void init_arr_0(float *r_buf_d, int al){

  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x;
  if(index < al){
    r_buf_d[index] = 0.0;
  }
}

/**communication for the multi gpu fluid called from host
 * @param *s_buf_d	Pointer to source device buffer
 * @param *r_buf_d	Pointer to receive device buffer
 * @param buf_size	buffer size
 * @param sn	      send node
 * @param rn      	receive node
*/
int cuda_comm::p2p_direct(float *s_buf_d, float *r_buf_d, int buf_size, int sn, int rn){
#if 1
  if(this_node == sn){

    //cudaEvent_t start_event, stop_event;
    //float time_memcpy;
    //int eventflags = cudaEventBlockingSync;
    //cuda_check_errors(cudaEventCreateWithFlags(&start_event, eventflags));
    //cuda_check_errors(cudaEventCreateWithFlags(&stop_event, eventflags));
    // P2P memcopy() benchmark
    //cuda_check_errors(cudaEventRecord(start_event, 0));
    // Enable peer access
    //printf("Enabling peer access between GPU%d and GPU%d...\n", 0, 1);
    cuda_check_errors(cudaSetDevice(0));
    cuda_check_errors(cudaDeviceEnablePeerAccess(1, 0));
    cuda_check_errors(cudaSetDevice(1));
    cuda_check_errors(cudaDeviceEnablePeerAccess(0, 0));

    cuda_check_errors(cudaMemcpy(r_buf_d, s_buf_d, buf_size, cudaMemcpyDefault));

    //cuda_check_errors(cudaEventRecord(stop_event, 0));
    //cuda_check_errors(cudaEventSynchronize(stop_event));
    //cuda_check_errors(cudaEventElapsedTime(&time_memcpy, start_event, stop_event));
    //printf("cudaMemcpyPeer / cudaMemcpy between GPU%d and GPU%d: %.2fGB/s\n", 0, 1,
    //      (1.0f / (time_memcpy / 1000.0f)) * ((100.0f * buf_size)) / 1024.0f / 1024.0f / 1024.0f);

  }
#endif
  return 1;
}

/**communication for the multi gpu fluid called from host
 * @param *s_buf_d	Pointer to source device buffer
 * @param *r_buf_d	Pointer to receive device buffer
 * @param buf_size	buffer size
 * @param sn	      send node
 * @param rn      	receive node
*/
int cuda_comm::p2p_direct_MPI(float *s_buf_d, float *r_buf_d, int buf_size, int sn, int rn){
  //p2p copy with cuda aware mpi (version >1.7)
//MPI rank 0
  MPI_Status status;
  int error_code;
  error_code = MPI_Sendrecv(s_buf_d, buf_size, MPI_FLOAT, sn, 101, r_buf_d, buf_size, MPI_FLOAT, rn, 101,
                   MPI_COMM_WORLD, &status);
    if (error_code != MPI_SUCCESS) {
      char error_string[BUFSIZ];
      int length_of_error_string, error_class;
      MPI_Error_class(error_code, &error_class);
      MPI_Error_string(error_class, error_string, &length_of_error_string);
      fprintf(stderr, "%3d: %s\n", this_node, error_string);
      MPI_Error_string(error_code, error_string, &length_of_error_string);
      fprintf(stderr, "%3d: %s\n", this_node, error_string);
      MPI_Abort(MPI_COMM_WORLD, error_code);
    }
  
  return 1;
}

/**communication for the multi gpu fluid called from host
 * @param *s_buf_h	Pointer to source host buffer
 * @param *r_buf_h	Pointer to receive host buffer
 * @param *s_buf_d	Pointer to source device buffer
 * @param *r_buf_d	Pointer to receive device buffer
 * @param buf_size	buffer size
 * @param sn	      send node
 * @param rn      	receive node
*/
int cuda_comm::p2p_indirect_MPI(float *s_buf_h, float *r_buf_h, float *s_buf_d, float *r_buf_d, int buf_size, int sn, int rn){

  //slowest but "always" available p2p copy
  MPI_Status status;
  // send node: copy of data from device to host and send it via MPI
#if 1
    cudaMemcpy(s_buf_h, s_buf_d, buf_size*sizeof(float), cudaMemcpyDeviceToHost);
    //for(int i=0; i<buf_size; ++i)
    //  printf("thisnode %i s_buf_h[%i]: %f \n", this_node, i, s_buf_h[i]);
#endif
    //sn: node which is send TO! and rn: node FROM which is received
    int error_code;
    error_code = MPI_Sendrecv(s_buf_h, buf_size, MPI_FLOAT, sn, 101, r_buf_h, buf_size, MPI_FLOAT, rn, 101,
                   MPI_COMM_WORLD, &status);
#if 1 
    //for(int i=0; i<buf_size; ++i)
    //  printf("thisnode %i r_buf_h[%i]: %f \n", this_node, i, r_buf_h[i]);
    cudaMemcpy(r_buf_d, r_buf_h, buf_size*sizeof(float), cudaMemcpyHostToDevice);
    //if any error ocours
#endif
    if (error_code != MPI_SUCCESS) {
      char error_string[BUFSIZ];
      int length_of_error_string, error_class;
      MPI_Error_class(error_code, &error_class);
      MPI_Error_string(error_class, error_string, &length_of_error_string);
      fprintf(stderr, "%3d: %s\n", this_node, error_string);
      MPI_Error_string(error_code, error_string, &length_of_error_string);
      fprintf(stderr, "%3d: %s\n", this_node, error_string);
      MPI_Abort(MPI_COMM_WORLD, error_code);
    }
  return 1;
}

/**send and receive the buffers for multi-GPU usage
 * @param s_buf_d pointer to send buffer of buffer IN the GPU memory
 * @param r_buf_d pointer to receive buffer of buffer IN the GPU memory
  */
int lbgpu::send_recv_buffer(float* s_buf_d, float* r_buf_d){

  LB_TRACE(printf("node %i sebd_recv_buffer gpu %i\n", this_node, lbdevicepar_gpu.gpu_number));
  //empty cpu buffers for communcation betwenn 2 gpus via cpu mem
  float *s_buf_h, *r_buf_h;
  int send_node, recv_node;
  unsigned offset;
  size_t buffer_size = 2*(size_of_buffer[0] + size_of_buffer[1] + size_of_buffer[2]);
  unsigned count[3] = {5*lbpar_gpu.number_of_halo_nodes[0], 5*lbpar_gpu.number_of_halo_nodes[1], 5*lbpar_gpu.number_of_halo_nodes[2]};
  s_buf_h = (float*)malloc(buffer_size);   
  r_buf_h = (float*)malloc(buffer_size);   
  //cudaMallocHost((void**)&s_buf_h, buffer_size);
  //cudaMallocHost((void**)&r_buf_h, buffer_size);
  //cudaHostAlloc((void**)&s_buf_h, buffer_size, cudaHostAllocMapped);
  //cudaHostAlloc((void**)&r_buf_h, buffer_size, cudaHostAllocMapped);
  //TODO  cuda_check_errors(cudaHostAlloc((void**)&plan[g].send_buffer_d, 6*sizeof(float*), cudaHostAllocMapped));   
  //    printf("thisnode %i node_grid: %i %i %i \n", this_node, node_grid[0], node_grid[1], node_grid[2]);
  /* send to right, recv from left i = 1, 7, 9, 11, 13 */
  send_node = node_neighbors[1];
  recv_node = node_neighbors[0];
  if (node_grid[0] > 1) {
    cuda_comm::p2p_indirect_MPI(s_buf_h, r_buf_h, s_buf_d, r_buf_d, count[0], send_node, recv_node);
    //printf("thisnode %i, :send_node: %i, recv_node: %i r_buf_h[0] %f\n",this_node, send_node, recv_node, r_buf_h[0]);
  } else {
    cudaMemcpy(r_buf_d,s_buf_d,size_of_buffer[0], cudaMemcpyDeviceToDevice);
   }
  /* send to left, recv from right i = 2, 8, 10, 12, 14 */
  send_node = node_neighbors[0];
  recv_node = node_neighbors[1];
    
  offset = 5*lbpar_gpu.number_of_halo_nodes[0];
  //printf("thisnode %i, offset %i, size_of_buffer[0] %i\n",this_node, offset, size_of_buffer[0]);
  if (node_grid[0] > 1) {
    cuda_comm::p2p_indirect_MPI((s_buf_h+offset), (r_buf_h+offset), (s_buf_d+offset), (r_buf_d+offset), count[0], send_node, recv_node);
  //printf("thisnode %i, :send_node: %i, recv_node: %i r_buf_h[0+offset] %f\n",this_node, send_node, recv_node, r_buf_h[0]);
  } else {
    cudaMemcpy((r_buf_d+offset),(s_buf_d+offset),size_of_buffer[0], cudaMemcpyDeviceToDevice);
   }
  /* send to front, recv from back i = 3, 7, 10, 15, 17 */
  send_node = node_neighbors[3];
  recv_node = node_neighbors[2];

  offset = 2*5*lbpar_gpu.number_of_halo_nodes[0];
  if (node_grid[1] > 1) {
    cuda_comm::p2p_indirect_MPI((s_buf_h+offset), (r_buf_h+offset), (s_buf_d+offset), (r_buf_d+offset), count[1], send_node, recv_node);
  } else {
    cudaMemcpy((r_buf_d+offset),(s_buf_d+offset),size_of_buffer[1], cudaMemcpyDeviceToDevice);
   }
  /* send to back, recv from front i = 4, 8, 9, 16, 18 */
  send_node = node_neighbors[2];
  recv_node = node_neighbors[3];
    
  offset = 5*(2*lbpar_gpu.number_of_halo_nodes[0] + lbpar_gpu.number_of_halo_nodes[1]);
  if (node_grid[1] > 1) {
    cuda_comm::p2p_indirect_MPI((s_buf_h+offset), (r_buf_h+offset), (s_buf_d+offset), (r_buf_d+offset), count[1], send_node, recv_node);
  } else {
    cudaMemcpy((r_buf_d+offset),(s_buf_d+offset),size_of_buffer[1], cudaMemcpyDeviceToDevice);
   }
  /* send to top, recv from bottom i = 5, 11, 14, 15, 18 */
  send_node = node_neighbors[5];
  recv_node = node_neighbors[4];
    
  offset = 5*2*(lbpar_gpu.number_of_halo_nodes[0] + lbpar_gpu.number_of_halo_nodes[1]);
  if (node_grid[2] > 1) {
    cuda_comm::p2p_indirect_MPI((s_buf_h+offset), (r_buf_h+offset), (s_buf_d+offset), (r_buf_d+offset), count[2], send_node, recv_node);
  } else {
    cudaMemcpy((r_buf_d+offset),(s_buf_d+offset),size_of_buffer[2], cudaMemcpyDeviceToDevice);
   }
  /* send to bottom, recv from top i = 6, 12, 13, 16, 17 */
  send_node = node_neighbors[4];
  recv_node = node_neighbors[5];
    
  offset = 5*2*(lbpar_gpu.number_of_halo_nodes[0] + lbpar_gpu.number_of_halo_nodes[1]) + 5*lbpar_gpu.number_of_halo_nodes[2];
  if (node_grid[2] > 1) {
    cuda_comm::p2p_indirect_MPI((s_buf_h+offset), (r_buf_h+offset), (s_buf_d+offset), (r_buf_d+offset), count[2], send_node, recv_node);
  } else {
    cudaMemcpy((r_buf_d+offset),(s_buf_d+offset),size_of_buffer[2], cudaMemcpyDeviceToDevice);
   }

  //printf("send_node: %i, recv_node: %i comm finished\n", send_node, recv_node);
  //printf("send_node: %i, recv_node: %i r_buf_h[0] %f\n", send_node, recv_node, r_buf_h[0]);
  lbgpu::cp_buffer_in_vd();
  free(s_buf_h);
  free(r_buf_h);

  return 1;
}

/**copy of the velocity densities from buffer into vd array
 * @param 
*/
void lbgpu::cp_buffer_in_vd(){

  LB_TRACE(printf("node %i cp_buffer_in_vd gpu %i\n", this_node, lbdevicepar_gpu.gpu_number));
  int threads_per_block = 64;
  int blocks_per_grid_y = 4;
  int blocks_per_grid_x = (lbpar_gpu.number_of_nodes + threads_per_block * blocks_per_grid_y - 1) /(threads_per_block * blocks_per_grid_y);
  dim3 dim_grid = make_uint3(blocks_per_grid_x, blocks_per_grid_y, 1);
  int g = 0; 
  //Attention GPU pointers
   // printf("node %i current pointer %p buffer %p\n", this_node, plan[g].current_nodes, plan[g].recv_buffer_d);
  KERNELCALL(write_buffer, dim_grid, threads_per_block, (*plan[g].current_nodes, plan[g].recv_buffer_d));

}

/**initialization for the lb gpu fluid called from host
 * @param *lbpar_gpu	Pointer to parameters to setup the lb field
*/
void lbgpu::init_GPU(LB_parameters_gpu *lbpar_gpu, LB_gpus *lbdevicepar_gpu){

  LB_TRACE(printf("node %i init_GPU gpu %i\n", this_node, lbdevicepar_gpu->gpu_number));
  LB_TRACE(printf("this_node: %i  local_box_l: %lf, %lf, %lf \n", this_node, local_box_l[0], local_box_l[1], local_box_l[2]));
  if (lbdevicepar_gpu->number_of_gpus == 1) {
    //dims stay like they are, just calc number of nodes 
    lbpar_gpu->number_of_nodes = (unsigned)(lbpar_gpu->dim_x*lbpar_gpu->dim_y*lbpar_gpu->dim_z);
    printf("Using only one GPU");
  }else{
    lbpar_gpu->dim_x = (unsigned)floor(local_box_l[0]/lbpar_gpu->agrid);
    lbpar_gpu->dim_y = (unsigned)floor(local_box_l[1]/lbpar_gpu->agrid);
    lbpar_gpu->dim_z = (unsigned)floor(local_box_l[2]/lbpar_gpu->agrid);
    lbpar_gpu->number_of_nodes_wo_halo = (unsigned) (lbpar_gpu->dim_x*lbpar_gpu->dim_y*lbpar_gpu->dim_z);
    //with halo in all three directions
    lbpar_gpu->dim_x += 2;
    lbpar_gpu->dim_y += 2;
    lbpar_gpu->dim_z += 2;
    //printf("dims: %u, %u, %u agrid %f\n", lbpar_gpu->dim_x, lbpar_gpu->dim_y, lbpar_gpu->dim_z, lbpar_gpu->agrid);
    lbpar_gpu->number_of_nodes = (unsigned) (lbpar_gpu->dim_x*lbpar_gpu->dim_y*lbpar_gpu->dim_z);
    //printf("init gpu number_of_nodes %i \n", lbpar_gpu->number_of_nodes);
    lbpar_gpu->number_of_halo_nodes[0] = (lbpar_gpu->dim_y*lbpar_gpu->dim_z);
    lbpar_gpu->number_of_halo_nodes[1] = (lbpar_gpu->dim_x*lbpar_gpu->dim_z);
    lbpar_gpu->number_of_halo_nodes[2] = (lbpar_gpu->dim_x*lbpar_gpu->dim_y);
    //printf("numberof_halonodes %i %i %i\n", lbpar_gpu->number_of_halo_nodes[0], lbpar_gpu->number_of_halo_nodes[1], lbpar_gpu->number_of_halo_nodes[2]);
  //
  }
  /** Allocate structs in device memory*/
  size_of_values = lbpar_gpu->number_of_nodes * sizeof(LB_values_gpu);
  size_of_values_wo_halo = lbpar_gpu->number_of_nodes_wo_halo * sizeof(LB_values_gpu);
  size_of_nodes_gpu = lbpar_gpu->number_of_nodes * 19 * sizeof(float);
  size_of_uint = lbpar_gpu->number_of_nodes * sizeof(unsigned int);
  size_of_3floats = lbpar_gpu->number_of_nodes * 3 * sizeof(float);
  stream = (cudaStream_t*)malloc(gpu_n*sizeof(cudaStream_t));
  //if(lbpar_gpu->number_of_nodes)
  //begin loop over devices i
  //multi gpu per mpi nodes stuff
  for(int g = 0; g < gpu_n; ++g){
    //set device i
    cuda_check_errors(cudaSetDevice(lbdevicepar_gpu->gpu_number));
#if 1
    if(plan[g].initflag){
      cudaFree(plan[g].device_values);
      cudaFree(plan[g].device_values_wo_halo);
      cudaFree(plan[g].nodes_a.vd);
      cudaFree(plan[g].nodes_b.vd);
      cudaFree(plan[g].nodes_a.seed);
      cudaFree(plan[g].nodes_b.seed);
      cudaFree(plan[g].nodes_a.boundary);
      cudaFree(plan[g].nodes_b.boundary);
      cudaFree(plan[g].node_f.force);
      cudaFree(plan[g].send_buffer_d);
      cudaFree(plan[g].recv_buffer_d);
    }
    cuda_check_errors(cudaDeviceReset());
    cuda_check_errors(cudaSetDeviceFlags(cudaDeviceMapHost));
#endif
    cuda_check_errors(cudaMalloc((void**)&plan[g].device_values, size_of_values));
    cuda_check_errors(cudaMalloc((void**)&plan[g].device_values_wo_halo, size_of_values_wo_halo));
#ifndef SHANCHEN
    cuda_check_errors(cudaMalloc((void**)&plan[g].nodes_a.vd, size_of_nodes_gpu));
    cuda_check_errors(cudaMalloc((void**)&plan[g].nodes_b.vd, size_of_nodes_gpu));
#else // SHANCHEN
    cuda_check_errors(cudaMalloc((void**)&plan[g].nodes_a.vd, SHANCHEN * size_of_nodes_gpu));
    cuda_check_errors(cudaMalloc((void**)&plan[g].nodes_b.vd, SHANCHEN * size_of_nodes_gpu));   
#endif // SHANCHEN
    cuda_check_errors(cudaMalloc((void**)&plan[g].nodes_a.seed, size_of_uint));
    cuda_check_errors(cudaMalloc((void**)&plan[g].nodes_a.boundary, size_of_uint));
    cuda_check_errors(cudaMalloc((void**)&plan[g].nodes_b.seed, size_of_uint));
    cuda_check_errors(cudaMalloc((void**)&plan[g].nodes_b.boundary, size_of_uint));
  
    cuda_check_errors(cudaMalloc((void**)&plan[g].node_f.force, size_of_3floats));

    //copy parameter into gpu const mem
    cuda_check_errors(cudaMemcpyToSymbol(para, lbpar_gpu, sizeof(LB_parameters_gpu)));
    cuda_check_errors(cudaMemcpyToSymbol(devpara, lbdevicepar_gpu, sizeof(LB_gpus)));
    //set up stream for execution order of gpu kernel functions
    cudaStreamCreate(&stream[g]);

    if (lbdevicepar_gpu->number_of_gpus > 1) {
      //size of different buffers
      size_of_buffer[0] = 5 * lbpar_gpu->number_of_halo_nodes[0] * sizeof(float); 
      size_of_buffer[1] = 5 * lbpar_gpu->number_of_halo_nodes[1] * sizeof(float); 
      size_of_buffer[2] = 5 * lbpar_gpu->number_of_halo_nodes[2] * sizeof(float); 
    //printf("number of halo nodes %i %i %i\n", lbpar_gpu->number_of_halo_nodes[0],lbpar_gpu->number_of_halo_nodes[1],lbpar_gpu->number_of_halo_nodes[2]);

    //allocate buffer in GPU mem
      cuda_check_errors(cudaMalloc((void**)&plan[g].send_buffer_d, 2*(size_of_buffer[0] + size_of_buffer[1] + size_of_buffer[2])));
      cuda_check_errors(cudaMalloc((void**)&plan[g].recv_buffer_d, 2*(size_of_buffer[0] + size_of_buffer[1] + size_of_buffer[2])));
      //new thread and block dims to ensure that enough threads are executed to init complete buffer
      int threads_per_block_b = 64;
      int blocks_per_grid_y_b = 4;
      int blocks_per_grid_x_b = ((5*2*(lbpar_gpu->number_of_halo_nodes[0]+lbpar_gpu->number_of_halo_nodes[1]+lbpar_gpu->number_of_halo_nodes[2])) + threads_per_block_b * blocks_per_grid_y_b - 1) /(threads_per_block_b * blocks_per_grid_y_b);
      dim3 dim_grid_b = make_uint3(blocks_per_grid_x_b, blocks_per_grid_y_b, 1);

      KERNELCALL(init_buf, dim_grid_b, threads_per_block_b, (plan[g].send_buffer_d, plan[g].recv_buffer_d));

    }
    //set flag to one for release of gpu memory 
    plan[g].initflag = 1;
  
    /** values for the kernel call */
    int threads_per_block = 64;
    int blocks_per_grid_y = 4;
    int blocks_per_grid_x = (lbpar_gpu->number_of_nodes + threads_per_block * blocks_per_grid_y - 1) /(threads_per_block * blocks_per_grid_y);
    dim3 dim_grid = make_uint3(blocks_per_grid_x, blocks_per_grid_y, 1);
  
    /** calc of veloctiydensities from given parameters and initialize the Node_Force array with zero */
    KERNELCALL(calc_n_equilibrium, dim_grid, threads_per_block, (plan[g].nodes_a));
    KERNELCALL(reinit_node_force, dim_grid, threads_per_block, (plan[g].node_f));
    KERNELCALL(reset_boundaries, dim_grid, threads_per_block, (plan[g].nodes_a, plan[g].nodes_b));

#ifdef SHANCHEN
    /* We must add compute values, shan-chen forces at this moment are zero as the densities are uniform*/
    KERNELCALL(values, dim_grid, threads_per_block, (plan[g].nodes_a, plan[g].device_values, plan[g].node_f));
#endif
    //set a nodes as current nodes due to equilibration values stored in there
    plan[g].current_nodes = &plan[g].nodes_a;
    //and use them in the first integration loop
    plan[g].intflag = 1;
    //printf("current pointer %p\n", plan[g].current_nodes->vd);
    //printf("init current pointer %p nodes a %p\n", plan[g].current_nodes, &plan[g].nodes_a);
    //printf("init send_buf %p recv_buf %p\n", plan[g].send_buffer_d, plan[g].recv_buffer_d);
    ///barrier for alle init kernels 
    //cudaStreamSynchronize(stream[g]);
    cuda_check_errors(cudaDeviceSynchronize());
    //cudaThreadSynchronize();
  }
}
/** reinitialization for the lb gpu fluid called from host
 * @param *lbpar_gpu	Pointer to parameters to setup the lb field
*/
void lbgpu::reinit_GPU(LB_parameters_gpu *lbpar_gpu, LB_gpus *lbdevicepar_gpu){

  LB_TRACE(printf("node %i reinit_GPU gpu %i\n", this_node, lbdevicepar_gpu->gpu_number));
  //begin loop over devices i
  for(int g = 0; g < gpu_n; ++g){
    //set device i
    cuda_check_errors(cudaSetDevice(lbdevicepar_gpu->gpu_number));
    /**write parameters in const memory*/
    cuda_check_errors(cudaMemcpyToSymbol(para, lbpar_gpu, sizeof(LB_parameters_gpu)));
    cuda_check_errors(cudaMemcpyToSymbol(devpara, lbdevicepar_gpu, sizeof(LB_gpus)));
  
    /** values for the kernel call */
    int threads_per_block = 64;
    int blocks_per_grid_y = 4;
    int blocks_per_grid_x = (lbpar_gpu->number_of_nodes + threads_per_block * blocks_per_grid_y - 1) /(threads_per_block * blocks_per_grid_y);
    dim3 dim_grid = make_uint3(blocks_per_grid_x, blocks_per_grid_y, 1);
  
    /** calc of veloctiydensities from given parameters and initialize the Node_Force array with zero */
    KERNELCALL(calc_n_equilibrium, dim_grid, threads_per_block, (plan[g].nodes_a));
  }
}

/**setup and call particle reallocation from the host
 * @param *lbpar_gpu	Pointer to parameters to setup the lb field
 * @param **host_data	Pointer to host information data
*/
void lbgpu::realloc_particle_GPU(LB_parameters_gpu *lbpar_gpu, LB_gpus *lbdevicepar_gpu, LB_particle_gpu **host_data){

  LB_TRACE(printf("node %i realloc_particle_GPU gpu %i\n", this_node, lbdevicepar_gpu->gpu_number));
  //begin loop over devices i
  for(int g = 0; g < gpu_n; ++g){
    //set device i
    cuda_check_errors(cudaSetDevice(lbdevicepar_gpu->gpu_number));
    cuda_check_errors(cudaMemcpyToSymbol(para, lbpar_gpu, sizeof(LB_parameters_gpu)));
    cuda_check_errors(cudaMemcpyToSymbol(devpara, lbdevicepar_gpu, sizeof(LB_gpus)));
    /** Allocate struct for particle positions */
    size_of_forces = lbdevicepar_gpu->number_of_particles * sizeof(LB_particle_force_gpu);
    size_of_positions = lbdevicepar_gpu->number_of_particles * sizeof(LB_particle_gpu);
    size_of_seed = lbdevicepar_gpu->number_of_particles * sizeof(LB_particle_seed_gpu);
    if(plan[g].partinitflag){
      cudaFreeHost(*host_data);
      cudaFree(plan[g].particle_force);
      cudaFree(plan[g].particle_data);
      cudaFree(plan[g].part);
    }
  #if !defined __CUDA_ARCH__ || __CUDA_ARCH__ >= 200
    /**pinned memory mode - use special function to get OS-pinned memory*/
    cudaHostAlloc((void**)host_data, size_of_positions, cudaHostAllocWriteCombined);
  #else
    cudaMallocHost((void**)host_data, size_of_positions);
  #endif
  
    if(plan[g].partinitflag){
      cudaFree(plan[g].particle_force);
      cudaFree(plan[g].particle_data);
      cudaFree(plan[g].part);
    }
    //cuda_check_errors(cudaMemcpyToSymbol(para, lbpar_gpu, sizeof(LB_parameters_gpu)));
  
    cuda_check_errors(cudaMalloc((void**)&plan[g].particle_force, size_of_forces));
    cuda_check_errors(cudaMalloc((void**)&plan[g].particle_data, size_of_positions));
    cuda_check_errors(cudaMalloc((void**)&plan[g].part, size_of_seed));
    plan[g].partinitflag  = 1;
    /** values for the particle kernel */
    int threads_per_block_particles = 64;
    int blocks_per_grid_particles_y = 4;
    int blocks_per_grid_particles_x = (lbdevicepar_gpu->number_of_particles + threads_per_block_particles * blocks_per_grid_particles_y - 1)/(threads_per_block_particles * blocks_per_grid_particles_y);
    dim3 dim_grid_particles = make_uint3(blocks_per_grid_particles_x, blocks_per_grid_particles_y, 1);
  
    if(lbdevicepar_gpu->number_of_particles) KERNELCALL(init_particle_force, dim_grid_particles, threads_per_block_particles, (plan[g].particle_force, plan[g].part));
  }
}
#ifdef LB_BOUNDARIES_GPU
/**setup and call boundaries from the host
 * @param *host_boundindex		Pointer to the host bound index
 * @param number_of_boundnodes	number of boundnodes
*/
void lbgpu::init_boundaries_GPU(int host_n_lb_boundaries, int number_of_boundnodes, int *host_boundary_node_list, int* host_boundary_index_list, float* host_lb_boundary_velocity){
  LB_TRACE(printf("node %i init_boundaries_GPU gpu %i\n", this_node, lbdevicepar_gpu.gpu_number));
  //begin loop over devices i
  for(int g = 0; g < gpu_n; ++g){
    //set device i
    cuda_check_errors(cudaSetDevice(lbdevicepar_gpu.gpu_number));
    int temp = host_n_lb_boundaries;
  //TODO
    size_of_boundindex = number_of_boundnodes*sizeof(int);
    cuda_check_errors(cudaMalloc((void**)&boundary_node_list, size_of_boundindex));
    cuda_check_errors(cudaMalloc((void**)&boundary_index_list, size_of_boundindex));
    cuda_check_errors(cudaMemcpy(boundary_index_list, host_boundary_index_list, size_of_boundindex, cudaMemcpyHostToDevice));
    cuda_check_errors(cudaMemcpy(boundary_node_list, host_boundary_node_list, size_of_boundindex, cudaMemcpyHostToDevice));
  
    cuda_check_errors(cudaMalloc((void**)&plan[g].lb_boundary_force   , 3*host_n_lb_boundaries*sizeof(float)));
    cuda_check_errors(cudaMalloc((void**)&plan[g].lb_boundary_velocity, 3*host_n_lb_boundaries*sizeof(float)));
    cuda_check_errors(cudaMemcpy(plan[g].lb_boundary_velocity, host_lb_boundary_velocity, 3*n_lb_boundaries*sizeof(float), cudaMemcpyHostToDevice));
    cuda_check_errors(cudaMemcpyToSymbol(n_lb_boundaries_gpu, &temp, sizeof(int)));
  
    /** values for the kernel call */
    int threads_per_block = 64;
    int blocks_per_grid_y = 4;
    int blocks_per_grid_x = (lbpar_gpu.number_of_nodes + threads_per_block * blocks_per_grid_y - 1) /(threads_per_block * blocks_per_grid_y);
    dim3 dim_grid = make_uint3(blocks_per_grid_x, blocks_per_grid_y, 1);
  
    KERNELCALL(reset_boundaries, dim_grid, threads_per_block, (plan[g].nodes_a, plan[g].nodes_b));
  
    if (n_lb_boundaries == 0) {
      cudaThreadSynchronize();
      return;
    }
    if(number_of_boundnodes == 0){
      fprintf(stderr, "WARNING: boundary cmd executed but no boundary node found!\n");
    } else{
      int threads_per_block_bound = 64;
      int blocks_per_grid_bound_y = 4;
      int blocks_per_grid_bound_x = (number_of_boundnodes + threads_per_block_bound * blocks_per_grid_bound_y - 1) /(threads_per_block_bound * blocks_per_grid_bound_y);
      dim3 dim_grid_bound = make_uint3(blocks_per_grid_bound_x, blocks_per_grid_bound_y, 1);
  
      KERNELCALL(init_boundaries, dim_grid_bound, threads_per_block_bound, 
                 (boundary_node_list, boundary_index_list, number_of_boundnodes, plan[g].nodes_a, plan[g].nodes_b));
    }
  
    cudaThreadSynchronize();
  }
}
#endif
/**setup and call extern single node force initialization from the host
 * @param *lbpar_gpu				Pointer to host parameter struct
*/
void lbgpu::reinit_extern_nodeforce_GPU(LB_parameters_gpu *lbpar_gpu, LB_gpus *lbdevicepar_gpu){

  LB_TRACE(printf("node %i reinit_extern_nodeforce_GPU gpu %i\n", this_node, lbdevicepar_gpu->gpu_number));
  //begin loop over devices i
  for(int g = 0; g < gpu_n; ++g){
    //set device i
    cuda_check_errors(cudaSetDevice(lbdevicepar_gpu->gpu_number));
    cuda_check_errors(cudaMemcpyToSymbol(para, lbpar_gpu, sizeof(LB_parameters_gpu)));
    cuda_check_errors(cudaMemcpyToSymbol(devpara, lbdevicepar_gpu, sizeof(LB_gpus)));
  
    /** values for the kernel call */
    int threads_per_block = 64;
    int blocks_per_grid_y = 4;
    int blocks_per_grid_x = (lbpar_gpu->number_of_nodes + threads_per_block * blocks_per_grid_y - 1) /(threads_per_block * blocks_per_grid_y);
    dim3 dim_grid = make_uint3(blocks_per_grid_x, blocks_per_grid_y, 1);
  
    KERNELCALL(reinit_node_force, dim_grid, threads_per_block, (plan[g].node_f));
  }
}
/**setup and call extern single node force initialization from the host
 * @param n_extern_nodeforces			number of nodes on which the external force has to be applied
 * @param *host_extern_nodeforces		Pointer to the host extern node forces
 * @param *lbpar_gpu				Pointer to host parameter struct
*/
void lbgpu::init_extern_nodeforces_GPU(int n_extern_nodeforces, LB_extern_nodeforce_gpu *host_extern_nodeforces, LB_parameters_gpu *lbpar_gpu, LB_gpus *lbdevicepar_gpu){
  LB_TRACE(printf("node %i init_extern_nodeforces_GPU gpu %i\n", this_node, lbdevicepar_gpu->gpu_number));

  //begin loop over devices i
  for(int g = 0; g < gpu_n; ++g){
    //set device i
    cuda_check_errors(cudaSetDevice(lbdevicepar_gpu->gpu_number));
    size_of_extern_nodeforces = n_extern_nodeforces*sizeof(LB_extern_nodeforce_gpu);
    cuda_check_errors(cudaMalloc((void**)&plan[g].extern_nodeforces, size_of_extern_nodeforces));
    cudaMemcpy(plan[g].extern_nodeforces, host_extern_nodeforces, size_of_extern_nodeforces, cudaMemcpyHostToDevice);
  
    if(lbpar_gpu->external_force == 0){
      cuda_check_errors(cudaMemcpyToSymbol(para, lbpar_gpu, sizeof(LB_parameters_gpu))); 
      cuda_check_errors(cudaMemcpyToSymbol(devpara, lbdevicepar_gpu, sizeof(LB_gpus))); 
    } 
    int threads_per_block_exf = 64;
    int blocks_per_grid_exf_y = 4;
    int blocks_per_grid_exf_x = (n_extern_nodeforces + threads_per_block_exf * blocks_per_grid_exf_y - 1) /(threads_per_block_exf * blocks_per_grid_exf_y);
    dim3 dim_grid_exf = make_uint3(blocks_per_grid_exf_x, blocks_per_grid_exf_y, 1);
  
    KERNELCALL(init_extern_nodeforces, dim_grid_exf, threads_per_block_exf, (n_extern_nodeforces, plan[g].extern_nodeforces, plan[g].node_f));
    cudaFree(plan[g].extern_nodeforces);
  }
}

/**setup and call particle kernel from the host
 * @param **host_data		Pointer to the host particle positions and velocities
*/
void lbgpu::particle_GPU(LB_particle_gpu *host_data){
  //begin loop over devices g
  LB_TRACE(printf("node %i particle_GPU gpu %i\n", this_node, lbdevicepar_gpu.gpu_number));
  for(int g = 0; g < gpu_n; ++g){
    //set device i
    cuda_check_errors(cudaSetDevice(lbdevicepar_gpu.gpu_number));
   /** get espresso md particle values*/
    cudaMemcpyAsync(plan[g].particle_data, host_data, size_of_positions, cudaMemcpyHostToDevice, stream[0]);
    /** call of the particle kernel */
    /** values for the particle kernel */
    int threads_per_block_particles = 64;
    int blocks_per_grid_particles_y = 4;
    int blocks_per_grid_particles_x = (lbdevicepar_gpu.number_of_particles + threads_per_block_particles * blocks_per_grid_particles_y - 1)/(threads_per_block_particles * blocks_per_grid_particles_y);
    dim3 dim_grid_particles = make_uint3(blocks_per_grid_particles_x, blocks_per_grid_particles_y, 1);
#ifndef SHANCHEN
    KERNELCALL(calc_fluid_particle_ia, dim_grid_particles, threads_per_block_particles, (*plan[g].current_nodes, plan[g].particle_data, plan[g].particle_force, plan[g].node_f, plan[g].part));
#else 
    KERNELCALL(calc_fluid_particle_ia, dim_grid_particles, threads_per_block_particles, (*plan[g].current_nodes, particle_data, particle_force, plan[g].node_f, part,device_values));
#endif
  }
}
/** setup and call kernel to copy particle forces to host
 * @param *host_forces contains the particle force computed on the GPU
*/
void lbgpu::copy_forces_GPU(LB_particle_force_gpu *host_forces){

  LB_TRACE(printf("node %i copy_forces_GPU gpu %i \n",this_node, lbdevicepar_gpu.gpu_number));
  //begin loop over devices g
  for(int g = 0; g < gpu_n; ++g){
    //set device i
    cuda_check_errors(cudaSetDevice(lbdevicepar_gpu.gpu_number));
    /** Copy result from device memory to host memory*/
    cudaMemcpy(host_forces, plan[g].particle_force, size_of_forces, cudaMemcpyDeviceToHost);
  
      /** values for the particle kernel */
    int threads_per_block_particles = 64;
    int blocks_per_grid_particles_y = 4;
    int blocks_per_grid_particles_x = (lbdevicepar_gpu.number_of_particles + threads_per_block_particles * blocks_per_grid_particles_y - 1)/(threads_per_block_particles * blocks_per_grid_particles_y);
    dim3 dim_grid_particles = make_uint3(blocks_per_grid_particles_x, blocks_per_grid_particles_y, 1);
  
    /** reset part forces with zero*/
    KERNELCALL(reset_particle_force, dim_grid_particles, threads_per_block_particles, (plan[g].particle_force));
  	
    //cudaStreamSynchronize(stream[g]);
    cudaThreadSynchronize();
  }
}

/** setup and call kernel for getting macroscopic fluid values of all nodes
 * @param *host_values struct to save the gpu values
*/
void lbgpu::get_values_GPU(LB_values_gpu *host_values){

  LB_TRACE(printf("node %i get_values_GPU gpu %i\n", this_node, lbdevicepar_gpu.gpu_number));
  //begin loop over devices g
  for(int g = 0; g < gpu_n; ++g){
    //set device g
    cuda_check_errors(cudaSetDevice(lbdevicepar_gpu.gpu_number));
    /** values for the kernel call */
    int threads_per_block = 64;
    int blocks_per_grid_y = 4;
    int blocks_per_grid_x = (lbpar_gpu.number_of_nodes + threads_per_block * blocks_per_grid_y - 1) /(threads_per_block * blocks_per_grid_y);
    dim3 dim_grid = make_uint3(blocks_per_grid_x, blocks_per_grid_y, 1);
#ifndef SHANCHEN
    if(lbdevicepar_gpu.number_of_gpus == 1){
      KERNELCALL(values, dim_grid, threads_per_block, (*plan[g].current_nodes, plan[g].device_values, plan[g].node_f));
    }else{
      KERNELCALL(values_wo_halo, dim_grid, threads_per_block, (*plan[g].current_nodes, plan[g].device_values_wo_halo, plan[g].node_f));
    }
#endif
    /* Note: in the Shan-Chen implementation the hydrodynamic fields (device_values) are computed in apply_forces(), 
  	   we need only to copy them
     */
    if(lbdevicepar_gpu.number_of_gpus == 1){
      cudaMemcpy(host_values, plan[g].device_values, size_of_values, cudaMemcpyDeviceToHost);
    }else{
      cudaMemcpy(host_values, plan[g].device_values_wo_halo, size_of_values_wo_halo, cudaMemcpyDeviceToHost);
    }
  }
}


/** setup and call kernel for getting macroscopic fluid values of all nodes
 * @param *host_values struct to save the gpu values
*/
void lbgpu::save_checkpoint_GPU(float *host_checkpoint_vd, unsigned int *host_checkpoint_seed, unsigned int *host_checkpoint_boundary, float *host_checkpoint_force){

  LB_TRACE(printf("node %i save cheackpoint gpu %i\n", this_node, lbdevicepar_gpu.gpu_number));
  //begin loop over devices g
  for(int g = 0; g < gpu_n; ++g){
    //set device i
    //printf("current pointer %p\n", plan[g].current_nodes->vd);
    cuda_check_errors(cudaSetDevice(lbdevicepar_gpu.gpu_number));
    cudaMemcpy(host_checkpoint_vd, plan[g].current_nodes->vd, size_of_nodes_gpu, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_checkpoint_seed, plan[g].current_nodes->seed, lbpar_gpu.number_of_nodes * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_checkpoint_boundary, plan[g].current_nodes->boundary, lbpar_gpu.number_of_nodes * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_checkpoint_force, plan[g].node_f.force, lbpar_gpu.number_of_nodes * 3 * sizeof(float), cudaMemcpyDeviceToHost);
  }
}
/** setup and call kernel for setting macroscopic fluid values of all nodes
   * @param *host_values struct to set stored values
   */
void lbgpu::load_checkpoint_GPU(float *host_checkpoint_vd, unsigned int *host_checkpoint_seed, unsigned int *host_checkpoint_boundary, float *host_checkpoint_force){

  LB_TRACE(printf("node %i load cheackpoint gpu %i\n", this_node, lbdevicepar_gpu.gpu_number));
  //begin loop over devices g
  for(int g = 0; g < gpu_n; ++g){
    //set device i
    cuda_check_errors(cudaSetDevice(lbdevicepar_gpu.gpu_number));
    cudaMemcpy(plan[g].current_nodes->vd, host_checkpoint_vd, size_of_nodes_gpu, cudaMemcpyHostToDevice);
      plan[g].intflag = 1;
    cudaMemcpy(plan[g].current_nodes->seed, host_checkpoint_seed, lbpar_gpu.number_of_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(plan[g].current_nodes->boundary, host_checkpoint_boundary, lbpar_gpu.number_of_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(plan[g].node_f.force, host_checkpoint_force, lbpar_gpu.number_of_nodes * 3 * sizeof(float), cudaMemcpyHostToDevice);
  }
}

/** get all the boundary flags for all nodes
 *  @param host_bound_array here go the values of the boundary flag
 */
void lbgpu::get_boundary_flags_GPU(unsigned int* host_bound_array){
   
  LB_TRACE(printf("node %i get_boundary_flags_GPU gpu %i\n", this_node, lbdevicepar_gpu.gpu_number));
  //begin loop over devices g
  for(int g = 0; g < gpu_n; ++g){
    //set device i
    cuda_check_errors(cudaSetDevice(lbdevicepar_gpu.gpu_number));
    unsigned int* device_bound_array;
    /** values for the kernel call */
    int threads_per_block = 64;
    int blocks_per_grid_y = 4;
    int blocks_per_grid_x = (lbpar_gpu.number_of_nodes + threads_per_block * blocks_per_grid_y - 1) /(threads_per_block * blocks_per_grid_y);
    dim3 dim_grid = make_uint3(blocks_per_grid_x, blocks_per_grid_y, 1);
    if(lbdevicepar_gpu.number_of_gpus ==1){
      cuda_check_errors(cudaMalloc((void**)&device_bound_array, lbpar_gpu.number_of_nodes*sizeof(unsigned int)));
      KERNELCALL(lb_get_boundaries, dim_grid, threads_per_block, (*plan[g].current_nodes, device_bound_array));
      cudaMemcpy(host_bound_array, device_bound_array, lbpar_gpu.number_of_nodes*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    }else{
      cuda_check_errors(cudaMalloc((void**)&device_bound_array, lbpar_gpu.number_of_nodes_wo_halo*sizeof(unsigned int)));
      KERNELCALL(lb_get_boundaries_wo_halo, dim_grid, threads_per_block, (*plan[g].current_nodes, device_bound_array));
      cudaMemcpy(host_bound_array, device_bound_array, lbpar_gpu.number_of_nodes_wo_halo*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    }
    cudaFree(device_bound_array);
  } 
}

/** setup and call kernel for getting macroscopic fluid values of a single node*/
void lbgpu::print_node_GPU(int single_nodeindex, LB_values_gpu *host_print_values){
  LB_TRACE(printf("node %i print_node_GPU gpu %i\n", this_node, lbdevicepar_gpu.gpu_number));
  //begin loop over devices g
  for(int g = 0; g < gpu_n; ++g){
    //set device g
    cuda_check_errors(cudaSetDevice(lbdevicepar_gpu.gpu_number));
  //TODO
    LB_values_gpu *device_print_values;
    cuda_check_errors(cudaMalloc((void**)&device_print_values, sizeof(LB_values_gpu)));
    int threads_per_block_print = 1;
    int blocks_per_grid_print_y = 1;
    int blocks_per_grid_print_x = 1;
    dim3 dim_grid_print = make_uint3(blocks_per_grid_print_x, blocks_per_grid_print_y, 1);
  
#ifndef SHANCHEN
    KERNELCALL(lb_print_node, dim_grid_print, threads_per_block_print, (single_nodeindex, device_print_values, *plan[g].current_nodes, plan[g].node_f));
#else 
    KERNELCALL(lb_print_node, dim_grid_print, threads_per_block_print, (single_nodeindex, device_print_values, *plan[g].current_nodes, plan[g].device_values));
#endif 
  //TODO
    cudaMemcpy(host_print_values, device_print_values, sizeof(LB_values_gpu), cudaMemcpyDeviceToHost);
  
    cudaFree(device_print_values);
  }
}
/** setup and call kernel to calculate the total momentum of the hole fluid
 * @param *mass value of the mass calcutated on the GPU
*/
void lbgpu::calc_fluid_mass_GPU(double* mass){

  LB_TRACE(printf("node %i calc_fluid_mass_GPU gpu %i\n", this_node, lbdevicepar_gpu.gpu_number));
  //begin loop over devices g
  for(int g = 0; g < gpu_n; ++g){
    //set device g
    cuda_check_errors(cudaSetDevice(lbdevicepar_gpu.gpu_number));
    float* tot_mass;
    float cpu_mass =  0.f ;
    cuda_check_errors(cudaMalloc((void**)&tot_mass, sizeof(float)));
    cudaMemcpy(tot_mass, &cpu_mass, sizeof(float), cudaMemcpyHostToDevice);
  
    /** values for the kernel call */
    int threads_per_block = 64;
    int blocks_per_grid_y = 4;
    int blocks_per_grid_x = (lbpar_gpu.number_of_nodes + threads_per_block * blocks_per_grid_y - 1) /(threads_per_block * blocks_per_grid_y);
    dim3 dim_grid = make_uint3(blocks_per_grid_x, blocks_per_grid_y, 1);
  
    KERNELCALL(calc_mass, dim_grid, threads_per_block,(*plan[g].current_nodes, tot_mass));
  
    cudaMemcpy(&cpu_mass, tot_mass, sizeof(float), cudaMemcpyDeviceToHost);
  
    cudaFree(tot_mass);
    mass[0] = (double)(cpu_mass);
  }
}

/** setup and call kernel to calculate the total momentum of the hole fluid
 *  @param host_mom value of the momentum calcutated on the GPU
 */
void lbgpu::calc_fluid_momentum_GPU(double* host_mom){
  LB_TRACE(printf("node %i calc_fluid_momentum_GPU gpu %i\n", this_node, lbdevicepar_gpu.gpu_number));
  //begin loop over devices g
  for(int g = 0; g < gpu_n; ++g){
    //set device g
    cuda_check_errors(cudaSetDevice(lbdevicepar_gpu.gpu_number));
  //TODO
    float* tot_momentum;
    float host_momentum[3] = { 0.f, 0.f, 0.f};
    cuda_check_errors(cudaMalloc((void**)&tot_momentum, 3*sizeof(float)));
    cudaMemcpy(tot_momentum, host_momentum, 3*sizeof(float), cudaMemcpyHostToDevice);
  
    /** values for the kernel call */
    int threads_per_block = 64;
    int blocks_per_grid_y = 4;
    int blocks_per_grid_x = (lbpar_gpu.number_of_nodes + threads_per_block * blocks_per_grid_y - 1) /(threads_per_block * blocks_per_grid_y);
    dim3 dim_grid = make_uint3(blocks_per_grid_x, blocks_per_grid_y, 1);
  
    KERNELCALL(momentum, dim_grid, threads_per_block,(*plan[g].current_nodes, tot_momentum, plan[g].node_f));
  
    cudaMemcpy(host_momentum, tot_momentum, 3*sizeof(float), cudaMemcpyDeviceToHost);
  
    cudaFree(tot_momentum);
    host_mom[0] = (double)(host_momentum[0]* lbpar_gpu.agrid/lbpar_gpu.tau);
    host_mom[1] = (double)(host_momentum[1]* lbpar_gpu.agrid/lbpar_gpu.tau);
    host_mom[2] = (double)(host_momentum[2]* lbpar_gpu.agrid/lbpar_gpu.tau);
  }
}
#ifndef SHANCHEN
//TODO check if the non shanchen version is still needed
/** setup and call kernel to calculate the temperature of the hole fluid
 *  @param host_temp value of the temperatur calcutated on the GPU
*/
void lbgpu::calc_fluid_temperature_GPU(double* host_temp){
  LB_TRACE(printf("node %i calc_fluid_temperature_GPU gpu %i\n", this_node, lbdevicepar_gpu.gpu_number));
  //begin loop over devices g
  for(int g = 0; g < gpu_n; ++g){
    //set device g
    cuda_check_errors(cudaSetDevice(lbdevicepar_gpu.gpu_number));
    //TODO
    float host_jsquared = 0.f;
    float* device_jsquared;
    cuda_check_errors(cudaMalloc((void**)&device_jsquared, sizeof(float)));
    cudaMemcpy(device_jsquared, &host_jsquared, sizeof(float), cudaMemcpyHostToDevice);
  
    /** values for the kernel call */
    int threads_per_block = 64;
    int blocks_per_grid_y = 4;
    int blocks_per_grid_x = (lbpar_gpu.number_of_nodes + threads_per_block * blocks_per_grid_y - 1) /(threads_per_block * blocks_per_grid_y);
    dim3 dim_grid = make_uint3(blocks_per_grid_x, blocks_per_grid_y, 1);
  
    KERNELCALL(temperature, dim_grid, threads_per_block,(*plan[g].current_nodes, device_jsquared));
  
    cudaMemcpy(&host_jsquared, device_jsquared, sizeof(float), cudaMemcpyDeviceToHost);
  
    host_temp[0] = (double)(host_jsquared*1./(3.f*lbpar_gpu.rho*lbpar_gpu.dim_x*lbpar_gpu.dim_y*lbpar_gpu.dim_z*lbpar_gpu.tau*lbpar_gpu.tau*lbpar_gpu.agrid));
  }
}
#endif
/** setup and call kernel to calculate the temperature of the hole fluid
 *  @param host_temp value of the temperatur calcutated on the GPU
*/
#ifdef SHANCHEN
void lbgpu::calc_fluid_temperature_GPU(double* host_temp){
  LB_TRACE(printf("node %i calc_fluid_temperature_GPU SHANCHEN gpu %i\n", this_node, lbdevicepar_gpu.gpu_number));
  //begin loop over devices g
  for(int g = 0; g < gpu_n; ++g){
    //set device g
    cuda_check_errors(cudaSetDevice(lbdevicepar_gpu.gpu_number));
    //TODO
    float host_jsquared = 0.f;
    float* device_jsquared;
    cuda_check_errors(cudaMalloc((void**)&device_jsquared, sizeof(float)));
    cudaMemcpy(device_jsquared, &host_jsquared, sizeof(float), cudaMemcpyHostToDevice);
  
    /** values for the kernel call */
    int threads_per_block = 64;
    int blocks_per_grid_y = 4;
    int blocks_per_grid_x = (lbpar_gpu.number_of_nodes + threads_per_block * blocks_per_grid_y - 1) /(threads_per_block * blocks_per_grid_y);
    dim3 dim_grid = make_uint3(blocks_per_grid_x, blocks_per_grid_y, 1);
  
    KERNELCALL(temperature, dim_grid, threads_per_block,(*plan[g].current_nodes, device_jsquared));
  
    cudaMemcpy(&host_jsquared, device_jsquared, sizeof(float), cudaMemcpyDeviceToHost);
    // SAW TODO: implement properly temperature calculation 
    *host_temp=0;
    #pragma unroll
    for(int ii=0;ii<SHANCHEN;++ii) { 
        *host_temp += (double)(host_jsquared*1./(3.f*lbpar_gpu.rho[ii]*lbpar_gpu.dim_x*lbpar_gpu.dim_y*lbpar_gpu.dim_z*lbpar_gpu.tau*lbpar_gpu.tau*lbpar_gpu.agrid));
    }
  }
}
#endif
/** setup and call kernel to get the boundary flag of a single node
 *  @param single_nodeindex number of the node to get the flag for
 *  @param host_flag her goes the value of the boundary flag
 */
void lbgpu::get_boundary_flag_GPU(int single_nodeindex, unsigned int* host_flag){
  LB_TRACE(printf("node %i get_bounday_flag_GPU gpu %i\n", this_node, lbdevicepar_gpu.gpu_number));
  //begin loop over devices g
  for(int g = 0; g < gpu_n; ++g){
    //set device i
    cuda_check_errors(cudaSetDevice(lbdevicepar_gpu.gpu_number));
  //TODO
    unsigned int* device_flag;
    cuda_check_errors(cudaMalloc((void**)&device_flag, sizeof(unsigned int)));
    int threads_per_block_flag = 1;
    int blocks_per_grid_flag_y = 1;
    int blocks_per_grid_flag_x = 1;
    dim3 dim_grid_flag = make_uint3(blocks_per_grid_flag_x, blocks_per_grid_flag_y, 1);
  
    KERNELCALL(lb_get_boundary_flag, dim_grid_flag, threads_per_block_flag, (single_nodeindex, device_flag, *plan[g].current_nodes));
  
    cudaMemcpy(host_flag, device_flag, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  
    cudaFree(device_flag);
  }
}


#ifdef SHANCHEN
/** set the density at a single node
 *  @param single_nodeindex the node to set the velocity for 
 *  @param host_velocity the velocity to set
 */
void lbgpu::set_node_rho_GPU(int single_nodeindex, float* host_rho){
   
  LB_TRACE(printf("node %i set_node_rho_GPU gpu %i \n",this_node, lbdevicepar_gpu->gpu_number));
  //begin loop over devices g
  for(int g = 0; g < gpu_n; ++g){
    //set device i
    cuda_check_errors(cudaSetDevice(lbdevicepar_gpu.gpu_number));
    float* device_rho;
    cuda_check_errors(cudaMalloc((void**)&device_rho, SHANCHEN*sizeof(float)));	
    cudaMemcpy(device_rho, host_rho, SHANCHEN*sizeof(float), cudaMemcpyHostToDevice);
    int threads_per_block_flag = 1;
    int blocks_per_grid_flag_y = 1;
    int blocks_per_grid_flag_x = 1;
    dim3 dim_grid_flag = make_uint3(blocks_per_grid_flag_x, blocks_per_grid_flag_y, 1);
    //TODO
    KERNELCALL(set_rho, dim_grid_flag, threads_per_block_flag, (*plan[g].current_nodes, plan[g].device_values, single_nodeindex, device_rho)); 
    cudaFree(device_rho);
  }
}
#endif // SHANCHEN
/** set the net velocity at a single node
 *  @param single_nodeindex the node to set the velocity for
 *  @param host_velocity the velocity to set
 */
void lbgpu::set_node_velocity_GPU(int single_nodeindex, float* host_velocity){

  LB_TRACE(printf("node %i set_node_velocity_GPU gpu %i \n",this_node, lbdevicepar_gpu.gpu_number));
  //begin loop over devices g
  for(int g = 0; g < gpu_n; ++g){
    //set device i
    cuda_check_errors(cudaSetDevice(lbdevicepar_gpu.gpu_number));
  //TODO
    float* device_velocity;
    cuda_check_errors(cudaMalloc((void**)&device_velocity, 3*sizeof(float)));
    cudaMemcpy(device_velocity, host_velocity, 3*sizeof(float), cudaMemcpyHostToDevice);
    int threads_per_block_flag = 1;
    int blocks_per_grid_flag_y = 1;
    int blocks_per_grid_flag_x = 1;
    dim3 dim_grid_flag = make_uint3(blocks_per_grid_flag_x, blocks_per_grid_flag_y, 1);
  
    KERNELCALL(set_u_equilibrium, dim_grid_flag, threads_per_block_flag, (*plan[g].current_nodes, single_nodeindex, device_velocity)); 
    cudaFree(device_velocity);
  }
}
/** reinit of params 
 * @param *lbpar_gpu struct containing the paramters of the fluid
*/
void lbgpu::reinit_parameters_GPU(LB_parameters_gpu *lbpar_gpu, LB_gpus *lbdevicepar_gpu){
  //begin loop over devices g
  LB_TRACE(printf("node %i reinit_parameters_GPU gpu %i \n",this_node, lbdevicepar_gpu->gpu_number));
  //printf("parameter gpu_n %i\n", gpu_n);
  for(int g = 0; g < gpu_n; ++g){
   //set device g
   cuda_check_errors(cudaSetDevice(lbdevicepar_gpu->gpu_number));
 
   /**write parameters in const memory*/
   cuda_check_errors(cudaMemcpyToSymbol(para, lbpar_gpu, sizeof(LB_parameters_gpu)));
   cuda_check_errors(cudaMemcpyToSymbol(devpara, lbdevicepar_gpu, sizeof(LB_gpus)));
  }
}
#ifdef LB_BOUNDARIES_GPU
void lbgpu::get_boundary_forces_GPU(double* forces) {
  LB_TRACE(printf("node %i get_boundary_forces_GPU gpu %i\n", this_node, lbdevicepar_gpu.gpu_number));
  //begin loop over devices g
  for(int g = 0; g < gpu_n; ++g){
    //set device g
    cuda_check_errors(cudaSetDevice(lbdevicepar_gpu.gpu_number));
    float* temp = (float*) malloc(3*n_lb_boundaries*sizeof(float));
    cuda_check_errors(cudaMemcpy(temp, plan[g].lb_boundary_force, 3*n_lb_boundaries*sizeof(float), cudaMemcpyDeviceToHost));
    for (int i =0; i<3*n_lb_boundaries; i++) {
      forces[i]=(double)temp[i];
    }
    free(temp);
  }
}
#endif
void lbgpu::barrier_GPU(){

  LB_TRACE(printf("node %i barrier_GPU gpu %i\n", this_node, lbdevicepar_gpu.gpu_number));
  for(int g = 0; g < gpu_n; ++g){
    //set device g
    cuda_check_errors(cudaSetDevice(lbdevicepar_gpu.gpu_number));
    printf("node %i gpu number %i\n", this_node, lbdevicepar_gpu.gpu_number);
    cuda_check_errors(cudaDeviceSynchronize());
  }

}

void lbgpu::send_recv_buffer_GPU(){

  LB_TRACE(printf("node %i send_recv_buffer_GPU gpu %i\n", this_node, lbdevicepar_gpu.gpu_number));
  for(int g = 0; g < gpu_n; ++g){
    //set device g
    cuda_check_errors(cudaSetDevice(lbdevicepar_gpu.gpu_number));
    lbgpu::send_recv_buffer(plan[g].send_buffer_d, plan[g].recv_buffer_d);
  }

}

/**integration kernel for the lb gpu fluid update called from host */
void lbgpu::integrate_GPU(){
  //begin loop over devices g
  //printf("integrate gpu_n %i\n", gpu_n);
  LB_TRACE(printf("node %i integrate_GPU gpu_number %i\n", this_node, lbdevicepar_gpu.gpu_number));
  for(int g = 0; g < gpu_n; ++g){
    //set device g
    cuda_check_errors(cudaSetDevice(lbdevicepar_gpu.gpu_number));
    
    /** values for the kernel call */
    int threads_per_block = 64;
    int blocks_per_grid_y = 4;
    int blocks_per_grid_x = (lbpar_gpu.number_of_nodes + threads_per_block * blocks_per_grid_y - 1) /(threads_per_block * blocks_per_grid_y);
    dim3 dim_grid = make_uint3(blocks_per_grid_x, blocks_per_grid_y, 1);
    //current_buffer = plan[g].vd_buffer;
  
    /**call of fluid step*/
    if (plan[g].intflag == 1){
      //printf("current pointer %p nodes a %p nodes b %p\n", plan[g].current_nodes, &plan[g].nodes_a, &plan[g].nodes_b);
      KERNELCALL(integrate, dim_grid, threads_per_block, (plan[g].nodes_a, plan[g].nodes_b, plan[g].device_values, plan[g].node_f, plan[g].send_buffer_d, &gpu_n));
      plan[g].current_nodes = &plan[g].nodes_b;
#ifdef LB_BOUNDARIES_GPU		
      if (n_lb_boundaries > 0) {
        KERNELCALL(bb_read, dim_grid, threads_per_block, (plan[g].nodes_a, plan[g].nodes_b, plan[g].lb_boundary_velocity, plan[g].lb_boundary_force));
        KERNELCALL(bb_write, dim_grid, threads_per_block, (plan[g].nodes_a, plan[g].nodes_b));
      }
#endif
     // printf("current pointer %p nodes b %p\n", plan[g].current_nodes, &plan[g].nodes_b);
    plan[g].intflag = 0;

    }else{
      KERNELCALL(integrate, dim_grid, threads_per_block, (plan[g].nodes_b, plan[g].nodes_a, plan[g].device_values, plan[g].node_f, plan[g].send_buffer_d, &gpu_n));
      plan[g].current_nodes = &plan[g].nodes_a;
#ifdef LB_BOUNDARIES_GPU		
      if (n_lb_boundaries > 0) {
        KERNELCALL(bb_read, dim_grid, threads_per_block, (plan[g].nodes_b, plan[g].nodes_a, plan[g].lb_boundary_velocity, plan[g].lb_boundary_force));
        KERNELCALL(bb_write, dim_grid, threads_per_block, (plan[g].nodes_b, plan[g].nodes_a));
      }
#endif
      //cudaThreadSynchronize();
      plan[g].intflag = 1;
    }
  } 
}

