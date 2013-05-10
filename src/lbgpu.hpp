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

/** \file lbgpu.h
 * Header file for lbgpu.c
 *
 * This is the header file for the Lattice Boltzmann implementation in lbgpu_cfile.c
 */

#ifndef LB_GPU_H
#define LB_GPU_H

/* For the D3Q19 model most functions have a separate implementation
 * where the coefficients and the velocity vectors are hardcoded
 * explicitly. This saves a lot of multiplications with 1's and 0's
 * thus making the code more efficient. */
#define D3Q19
#define LBQ 19

/** \name Parameter fields for Lattice Boltzmann
 * The numbers are referenced in \ref mpi_bcast_lb_params
 * to determine what actions have to take place upon change
 * of the respective parameter. */
/*@{*/
#define LBPAR_DENSITY   0 /**< fluid density */
#define LBPAR_VISCOSITY 1 /**< fluid kinematic viscosity */
#define LBPAR_AGRID     2 /**< grid constant for fluid lattice */
#define LBPAR_TAU       3 /**< time step for fluid propagation */
#define LBPAR_FRICTION  4 /**< friction coefficient for viscous coupling between particles and fluid */
#define LBPAR_EXTFORCE  5 /**< external force acting on the fluid */
#define LBPAR_BULKVISC  6 /**< fluid bulk viscosity */
#ifdef CONSTRAINTS
#define LBPAR_BOUNDARY  7 /**< boundary parameters */
#endif
#ifdef SHANCHEN
#define LBPAR_COUPLING 8
#define LBPAR_MOBILITY 9
#endif
/*@}*/

/**-------------------------------------------------------------------------*/
/** Data structure holding the parameters for the Lattice Boltzmann system for gpu. */
struct LB_parameters_gpu
{
  
#ifndef SHANCHEN
  /** number density (LJ units) */
  float rho;
  /** mu (LJ units) */
  float mu;
  /*viscosity (LJ) units */
  float viscosity;
  /** relaxation rate of shear modes */
  float gamma_shear;
  /** relaxation rate of bulk modes */
  float gamma_bulk;
  /**      */
  float gamma_odd;
  float gamma_even;
  /** friction coefficient for viscous coupling (LJ units)
   * Note that the friction coefficient is quite high and may
   * lead to numerical artifacts with low order integrators */
  float friction;
  /** amplitude of the fluctuations in the viscous coupling */
  float lb_coupl_pref;
  float lb_coupl_pref2;
  float bulk_viscosity;
#else //SHANCHEN
  /** number density (LJ units) */
  float rho[SHANCHEN];
  /** mobility. They are actually SHANCHEN-1 in number, we leave SHANCHEN here for practical reasons*/
  float gamma_mobility[SHANCHEN];
  float mobility[SHANCHEN];
#if ( SHANCHEN == 1 )
  float coupling[2];
#else  // SHANCHEN == 1 
  float coupling[SHANCHEN*SHANCHEN];
#endif   // SHANCHEN == 1 
  /** mu (LJ units) */
  float mu[SHANCHEN];
  /*viscosity (LJ) units */
  float viscosity[SHANCHEN];
  /** relaxation rate of shear modes */
  float gamma_shear[SHANCHEN];
  /** relaxation rate of bulk modes */
  float gamma_bulk[SHANCHEN];
  /**      */
  float gamma_odd[SHANCHEN];
  float gamma_even[SHANCHEN];
  /** friction coefficient for viscous coupling (LJ units)
   * Note that the friction coefficient is quite high and may
   * lead to numerical artifacts with low order integrators */
  float friction[SHANCHEN];
  /** amplitude of the fluctuations in the viscous coupling */
  float lb_coupl_pref[SHANCHEN];
  float lb_coupl_pref2[SHANCHEN];
  float bulk_viscosity[SHANCHEN];

#endif //SHANCHEN
  /** lattice spacing (LJ units) */
  float agrid;

  /** time step for fluid propagation (LJ units)
   *  Note: Has to be larger than MD time step! */
  float tau;

  /** MD tiemstep */
  float time_step;

  unsigned int dim_x;
  unsigned int dim_y;
  unsigned int dim_z;
  unsigned int local_box_l[3];
  int gpu_number;
  int number_of_gpus;
  int cpus_per_gpu;
  int gpus_per_cpu;

  unsigned int number_of_nodes;
  unsigned number_of_halo_nodes[3];

  unsigned int number_of_particles;
  /** Flag indicating whether fluctuations are present. */
  int fluct;
  /**to calc and print out phys values */
  int calc_val;

  int external_force;

  float ext_force[3];

  unsigned int your_seed;

  unsigned int reinit;

  //constructor
  LB_parameters_gpu(){};
  //LB_parameters_gpu(float _tau, int _reinit = 1) : tau(_tau), reinit(_reinit){};
//  LB_parameters_gpu(unsigned int _reinit, float rho, float mu, float viscosity, float gamma_shear, float gamma_bulk, float gamma_odd, float gamma_even, float friction, float agrid, float tau, float time_step, unsigned int dim_x, unsigned int dim_y, unsigned int dim_z, int gpu_number, int number_of_gpus, int cpus_per_gpu, int gpus_per_cpu, unsigned int number_of_nodes, unsigned int number_of_particles, int fluct, int calc_val, int external_force, unsigned int your_seed) : reinit(_reinit), rho(1.0), mu(1.0), viscosity(1.0), gamma_shear(1.0), gamma_bulk(1.0), gamma_odd(0.0), gamma_even(0.0), friction(1.0), agrid(1.0), tau(0.01), time_step(0.01), dim_x(12), dim_y(12), dim_z(12), gpu_number(0), number_of_gpus(0), cpus_per_gpu(0), gpus_per_cpu(0), number_of_nodes(0), number_of_particles(0), fluct(0), calc_val(0), external_force(0), your_seed(12345) {};

};
/** Data structure holding the phys. values for the Lattice Boltzmann system. */
typedef struct {

#ifndef SHANCHEN
  /** density of the node */
  float rho;
#else // SHANCHEN
  float rho[SHANCHEN];
#endif // SHANCHEN
  /** veolcity of the node */
  float v[3];


  /** stresstensor of the node */
  float pi[6];

} LB_values_gpu;
/** Data structure holding the velocitydensities for the Lattice Boltzmann system. */
typedef struct {

  /** velocitydensity of the node */
  float *vd;
  /** seed for the random gen */
  unsigned int *seed;
  /** flag indicating whether this site belongs to a boundary */
  unsigned int *boundary;

} LB_nodes_gpu;
/** Data structure for the randomnr and the seed. */
typedef struct {

  float randomnr[2];

  unsigned int seed;

} LB_randomnr_gpu;
/** Data structure for particle force. */
typedef struct {
  /** force on the particle given to md part */
  float f[3];

} LB_particle_force_gpu;
/** Data structure for particle pos, vel, ... */
typedef struct {
  /** particle position given from md part*/
  float p[3];
  /** particle momentum struct velocity p.m->v*/
  float v[3];
#ifdef SHANCHEN
  float solvation[2*SHANCHEN];
#endif 
#ifdef LB_ELECTROHYDRODYNAMICS
  float mu_E[3];
#endif
  unsigned int fixed;

} LB_particle_gpu;
/** Data structure for node force */
typedef struct {

  float *force;

} LB_node_force_gpu;
/** Data structure for external node force */
typedef struct {

  float force[3];

  unsigned int index;

} LB_extern_nodeforce_gpu;
/** Data structure for seed/state of rng of nodes */
typedef struct {

  unsigned int seed;

} LB_particle_seed_gpu;
/** Data structure for node structs of different gpus */
typedef struct {
  //structs for nodes    
  LB_values_gpu *device_values;
  /** structs for velocity densities */
  LB_nodes_gpu nodes_a;
  LB_nodes_gpu nodes_b;
  LB_nodes_gpu *current_nodes;
  /** struct for particle force */
  LB_particle_force_gpu *particle_force;
  /** struct for particle position and veloctiy */
  LB_particle_gpu *particle_data;
  /** struct for node force */
  LB_node_force_gpu node_f;
  /** struct for storing particle rn seed */
  LB_particle_seed_gpu *part;

  LB_extern_nodeforce_gpu *extern_nodeforces;
  //intflag for double buffering method
  unsigned int intflag;
  //initflags for release of memory
  unsigned int initflag;
  unsigned int partinitflag;
 
  float *lb_boundary_force;
  float *lb_boundary_velocity;

  //Stream for asynchronous command execution
  float *send_buffer_d;
  float *recv_buffer_d;
  //cudaStream_t stream;
  
} plan_gpu;

/************************************************************/
/** \name Exported Variables */
/************************************************************/
/*@{*/

/** 
 */
//#ifdef __cplusplus
//extern "C" {
//#endif
/** Switch indicating momentum exchange between particles and fluid */
extern LB_parameters_gpu lbpar_gpu;
extern LB_values_gpu *host_values;
extern int transfer_momentum_gpu;
extern LB_extern_nodeforce_gpu *extern_nodeforces_gpu;
extern int n_lb_boundaries;

//MPI stuffi
#if 0
extern int node_grid[3];
extern int this_node;
extern int n_nodes;
extern MPI_Comm comm_cart;
extern int node_pos[3];
extern int node_neighbors[6];
#endif


/*@}*/

/**namespaces */
/** hardware info */
namespace hw {
  void get_dev_count();
  void set_dev(int dev);
  void check_dev(int dev);
}

/**namespaces */
/** mutli gpu cuda communication */
namespace cuda_comm {
  int p2p_direct(float *s_buf_d, float *r_buf_d, int buf_size, int sn, int rn);
  int p2p_direct_MPI(float *s_buf_d, float *r_buf_d, int buf_size, int sn, int rn);
  int p2p_indirect_MPI(float *s_buf_h, float *r_buf_h,float *s_buf_d, float *r_buf_d, int buf_size, int sn, int rn);
}

/**Namespace for lbgpu functions
 * @param 
*/
/*-------------------------------------------------------*/
namespace lbgpu {
  /************************************************************/
  /** \name Exported Functions */
  /************************************************************/
  /*@{*/
  /** function called in lb.c only to get params from tcl */
  void params_change(int field);
  
  /**lb update function called by integrate.cpp
   * */
  void lattice_boltzmann_update();
  /** lb update function calls integrate on gpu 
   * */
  void update();
  
  /** (Pre-)initializes data structures. */
  void pre_init();
  
  /** Performs a full initialization of
   *  the Lattice Boltzmann system. All derived parameters
   *  and the fluid are reset to their default values. */
  void init();
  
  /** (Re-)initializes the derived parameters
   *  for the Lattice Boltzmann system.
   *  The current state of the fluid is unchanged. */
  void reinit_parameters();
  
  /** (Re-)initializes the fluid. */
  void reinit_fluid();
  
  /** Resets the forces on the fluid nodes */
  //void lb_reinit_forces();
  
  /** (Re-)initializes the particle array*/
  void realloc_particles();

  /** calc particle fluid interaction*/
  void calc_particle_lattice_ia();

  /** sen calculated particle forces from GPU to CPU and distibute them*/
  void send_forces();
  
  void init_GPU(LB_parameters_gpu *lbpar_gpu);
  void integrate_GPU();
  void particle_GPU(LB_particle_gpu *host_data);
  #ifdef SHANCHEN
  void calc_shanchen_GPU();
  #endif
  void free_GPU();
  void get_values_GPU(LB_values_gpu *host_values);
  void realloc_particle_GPU(LB_parameters_gpu *lbpar_gpu, LB_particle_gpu **host_data);
  void copy_forces_GPU(LB_particle_force_gpu *host_forces);
  void print_node_GPU(int single_nodeindex, LB_values_gpu *host_print_values);
  #ifdef LB_BOUNDARIES_GPU
  void init_boundaries_GPU(int n_lb_boundaries, int number_of_boundnodes, int* host_boundary_node_list, int* host_boundary_index_list, float* lb_bounday_velocity);
  #endif
  void init_extern_nodeforces_GPU(int n_extern_nodeforces, LB_extern_nodeforce_gpu *host_extern_nodeforces, LB_parameters_gpu *lbpar_gpu);
  
  void calc_fluid_mass_GPU(double* mass);
  void calc_fluid_momentum_GPU(double* host_mom);
  void calc_fluid_temperature_GPU(double* host_temp);

  void get_boundary_flag_GPU(int single_nodeindex, unsigned int* host_flag);
  void get_boundary_flags_GPU(unsigned int* host_bound_array);
  
  void set_node_velocity_GPU(int single_nodeindex, float* host_velocity);
  void set_node_rho_GPU(int single_nodeindex, float* host_rho);
  
  void reinit_parameters_GPU(LB_parameters_gpu *lbpar_gpu);
  void reinit_extern_nodeforce_GPU(LB_parameters_gpu *lbpar_gpu);
  void reinit_GPU(LB_parameters_gpu *lbpar_gpu);

  int lbnode_set_extforce_GPU(int ind[3], double f[3]);
  void get_boundary_forces_GPU(double* forces);

  void save_checkpoint_GPU(float *host_checkpoint_vd, unsigned int *host_checkpoint_seed, unsigned int *host_checkpoint_boundary, float *host_checkpoint_force);
  void load_checkpoint_GPU(float *host_checkpoint_vd, unsigned int *host_checkpoint_seed, unsigned int *host_checkpoint_boundary, float *host_checkpoint_force);
  /** init functions
   * */
  void init_struct();
  void setup_plan();
  void reinit_plan(int* dev, int count);

  /**multi-gpu functions
   * */
  int send_recv_buffer(float* s_buf_d, float* r_buf_d);
  void cp_buffer_in_vd();
  int set_devices(int* devices, int count);
  int get_devices(int* devices);
  /**no used but still present function :p
   */
  void release();

}
/*@{*/

#endif /* LB_GPU_H */
