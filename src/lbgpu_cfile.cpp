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
#include <mpi.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include "lbgpu.hpp"
#include "utils.hpp"
#include "communication.hpp"
#include "thermostat.hpp"
#include "grid.hpp"
#include "domain_decomposition.hpp"
#include "integrate.hpp"
#include "interaction_data.hpp"
#include "particle_data.hpp"
#include "global.hpp"
#include "lb-boundaries.hpp"
#ifdef LB_GPU

/** Action number for \ref mpi_get_particles. */
#define REQ_GETPARTS  16
#ifndef D3Q19
#error The implementation only works for D3Q19 so far!
#endif

/** Struct holding the Lattice Boltzmann parameters */
LB_parameters_gpu lbpar_gpu;
LB_gpus lbdevicepar_gpu;
//= { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0 ,0.0, -1.0, 0, 0, 0, 0, 0, 0, 1, 0, {0.0, 0.0, 0.0}, 12345, 0};
LB_values_gpu *host_values = NULL;
LB_nodes_gpu *host_nodes = NULL;
LB_particle_force_gpu *host_forces = NULL;
LB_particle_gpu *host_data = NULL;

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

static void mpi_get_particles_lb(LB_particle_gpu *host_result);
static void mpi_get_particles_slave_lb();
static void mpi_send_forces_lb(LB_particle_force_gpu *host_forces);
static void mpi_send_forces_slave_lb();

int n_extern_nodeforces = 0;
LB_extern_nodeforce_gpu *host_extern_nodeforces = NULL;

/*-----------------------------------------------------------*/
/** main of lb_gpu_programm */
/*-----------------------------------------------------------*/
/** lattice boltzmann update gpu called from integrate.c
*/
void lbgpu::lattice_boltzmann_update() {

  int factor = (int)round(lbpar_gpu.tau/time_step);

  fluidstep += 1;

  if (fluidstep>=factor) {
    fluidstep=0;

    //lbgpu::integrate_GPU();
#if 0
  if(this_node == 0)
  lb_lbfluid_save_checkpoint("checkpoint2.dat", 0);
  else
  lb_lbfluid_save_checkpoint("checkpoint3.dat", 0);
#endif
    if (lbdevicepar_gpu.number_of_gpus > 1) {
      lbgpu::integrate_multigpu_GPU();
      lbgpu::send_recv_buffer_gpu();
      lbgpu::bb_bounds_GPU();
    }else{
    lbgpu::integrate_GPU();
    }
#if 0
  if(this_node == 0)
  lb_lbfluid_save_checkpoint("checkpoint4.dat", 0);
  else
  lb_lbfluid_save_checkpoint("checkpoint5.dat", 0);
#endif
    LB_TRACE (fprintf(stderr,"node %i lb_integrate_GPU finished\n", this_node));

  }
}

/** Calculate particle lattice interactions called from forces.c
*/
void lbgpu::calc_particle_lattice_ia() {

  if (transfer_momentum_gpu) {
    mpi_get_particles_lb(host_data);

    if(this_node == 0){
#if 0
      for (i=0;i<n_total_particles;i++) {
        fprintf(stderr, "%i particle posi: , %f %f %f\n", i, host_data[i].p[0], host_data[i].p[1], host_data[i].p[2]);
      }
#endif

      if(lbdevicepar_gpu.number_of_particles){ lbgpu::particle_GPU(host_data);

        LB_TRACE (fprintf(stderr,"node %i lb_calc_particle_lattice_ia_gpu \n",this_node));
      }
    }
  }
}

/**copy forces from gpu to cpu and call mpi routines to add forces to particles
*/
void lbgpu::send_forces(){

  if (transfer_momentum_gpu) {
    if(this_node == 0){
      if (lbdevicepar_gpu.number_of_particles){
        lbgpu::copy_forces_GPU(host_forces);

        LB_TRACE (fprintf(stderr,"node %i send_forces \n", this_node));
      }
#if 0
        for (i=0;i<n_total_particles;i++) {
          fprintf(stderr, "%i particle forces , %f %f %f \n", i, host_forces[i].f[0], host_forces[i].f[1], host_forces[i].f[2]);
        }
#endif
    }
    mpi_send_forces_lb(host_forces);
  }
}

/** (re-) allocation of the memory need for the particles (cpu part)*/
void lbgpu::realloc_particles(){

  lbdevicepar_gpu.number_of_particles = n_total_particles;
  LB_TRACE (printf("node %i #particles realloc\t %u \n", this_node,lbdevicepar_gpu.number_of_particles));
  //fprintf(stderr, "%u \t \n", lbpar_gpu.number_of_particles);
  /**-----------------------------------------------------*/
  /** allocating of the needed memory for several structs */
  /**-----------------------------------------------------*/
  /**Allocate struct for particle forces */
  size_t size_of_forces = lbdevicepar_gpu.number_of_particles * sizeof(LB_particle_force_gpu);
  host_forces = (LB_particle_force_gpu*) realloc(host_forces, size_of_forces);

  lbpar_gpu.your_seed = (unsigned int)i_random(max_ran);

  LB_TRACE (fprintf(stderr,"test your_seed %u \n", lbpar_gpu.your_seed));
  lbgpu::realloc_particle_GPU(&lbpar_gpu,  &lbdevicepar_gpu, &host_data);
}
/** (Re-)initializes the fluid according to the given value of rho. */
void lbgpu::reinit_fluid() {

  //lbpar_gpu.your_seed = (unsigned int)i_random(max_ran);
  lbgpu::reinit_parameters();
  if(lbpar_gpu.number_of_global_nodes != 0){
    //printf("node %i number_of_nodes %i\n",this_node, lbpar_gpu.number_of_nodes);
    lbgpu::reinit_GPU(&lbpar_gpu, &lbdevicepar_gpu);
    lbpar_gpu.reinit = 1;
  
#if 0
  if(this_node == 0)
  lb_lbfluid_save_checkpoint("checkpoint6.dat", 0);
  else
  lb_lbfluid_save_checkpoint("checkpoint7.dat", 0);
  LB_TRACE (fprintf(stderr,"lb_reinit_fluid_gpu finished\n"));
#endif
  }
}

/** Release the fluid. */
/*not needed in Espresso but still not deleted*/
void lbgpu::release(){

  free(host_nodes);
  free(host_values);
  free(host_forces);
  free(host_data);
}

/** call excange of buffer after mpi barrier */
/*not needed in Espresso but still not deleted*/
void lbgpu::send_recv_buffer_gpu(){

  lbgpu::barrier_GPU();

  MPI_Barrier(comm_cart);
  
  lbgpu::send_recv_buffer_GPU();
  printf("node %i send_recv finished\n",this_node);

}
/** inint parameter strcut with default values */
/**/
void lbgpu::init_struct(){

 lbpar_gpu.rho=1.0;
 lbpar_gpu.mu=1.0;
 lbpar_gpu.viscosity=1.0;
 lbpar_gpu.gamma_shear=0.0;
 lbpar_gpu.gamma_bulk=0.0;
 lbpar_gpu.gamma_odd=0.0;
 lbpar_gpu.gamma_even=0.0;
 lbpar_gpu.agrid=1.0;
 lbpar_gpu.tau=0.01;
 lbpar_gpu.friction=5.0;
 lbpar_gpu.time_step=0.01;
 lbpar_gpu.lb_coupl_pref=0.0;
 lbpar_gpu.lb_coupl_pref2=0.0;
 lbpar_gpu.bulk_viscosity=0.0;
 lbpar_gpu.dim_x=0;
 lbpar_gpu.dim_y=0;
 lbpar_gpu.dim_z=0;
 //for(int i=0;i<3;++i)
 //  lbpar_gpu.local_box_l[i]=0.0;
 lbdevicepar_gpu.gpu_number=-1;
 lbdevicepar_gpu.number_of_gpus=1;
 lbdevicepar_gpu.cpus_per_gpu=-1;
 lbdevicepar_gpu.gpus_per_cpu=-1;
 lbpar_gpu.number_of_nodes=0;
 lbdevicepar_gpu.number_of_particles=0;
 lbpar_gpu.fluct=0;
 lbpar_gpu.calc_val=1;
 lbpar_gpu.external_force=0;
 for(int i=0;i<3;++i) *(lbpar_gpu.ext_force+i) = 0;
}

/** (Re-)initializes the fluid. */
void lbgpu::reinit_parameters() {

  lbpar_gpu.mu = 0.0;
  lbpar_gpu.time_step = (float)time_step;

  if (lbpar_gpu.viscosity > 0.0) {
    /* Eq. (80) Duenweg, Schiller, Ladd, PRE 76(3):036704 (2007). */
    lbpar_gpu.gamma_shear = 1. - 2./(6.*lbpar_gpu.viscosity*lbpar_gpu.tau/(lbpar_gpu.agrid*lbpar_gpu.agrid) + 1.);   
  }

  if (lbpar_gpu.bulk_viscosity > 0.0) {
    /* Eq. (81) Duenweg, Schiller, Ladd, PRE 76(3):036704 (2007). */
    lbpar_gpu.gamma_bulk = 1. - 2./(9.*lbpar_gpu.bulk_viscosity*lbpar_gpu.tau/(lbpar_gpu.agrid*lbpar_gpu.agrid) + 1.);
  }

  if (temperature > 0.0) {  /* fluctuating hydrodynamics ? */

    lbpar_gpu.fluct = 1;
	LB_TRACE (fprintf(stderr, "fluct on \n"));
    /* Eq. (51) Duenweg, Schiller, Ladd, PRE 76(3):036704 (2007).*/
    /* Note that the modes are not normalized as in the paper here! */

    lbpar_gpu.mu = (float)temperature/c_sound_sq*lbpar_gpu.tau*lbpar_gpu.tau/(lbpar_gpu.agrid*lbpar_gpu.agrid);
    //lbpar_gpu->mu *= agrid*agrid*agrid;  // Marcello's conjecture

    /* lb_coupl_pref is stored in MD units (force)
     * Eq. (16) Ahlrichs and Duenweg, JCP 111(17):8225 (1999).
     * The factor 12 comes from the fact that we use random numbers
     * from -0.5 to 0.5 (equally distributed) which have variance 1/12.
     * time_step comes from the discretization.
     */

    lbpar_gpu.lb_coupl_pref = sqrt(12.f*2.f*lbpar_gpu.friction*(float)temperature/lbpar_gpu.time_step);
    lbpar_gpu.lb_coupl_pref2 = sqrt(2.f*lbpar_gpu.friction*(float)temperature/lbpar_gpu.time_step);

  } else {
    /* no fluctuations at zero temperature */
    lbpar_gpu.fluct = 0;
    lbpar_gpu.lb_coupl_pref = 0.0;
    lbpar_gpu.lb_coupl_pref2 = 0.0;
  }
  //lbpar_gpu.local_box_l[0] = (unsigned) local_box_l[0];
  //lbpar_gpu.local_box_l[1] = (unsigned) local_box_l[1];
  //lbpar_gpu.local_box_l[2] = (unsigned) local_box_l[2];
	LB_TRACE (fprintf(stderr,"node %i lb_reinit_parameters_gpu \n", this_node));
  lbgpu::reinit_parameters_GPU(&lbpar_gpu, &lbdevicepar_gpu);
}

/** Performs a full initialization of
 *  the Lattice Boltzmann system. All derived parameters
 *  and the fluid are reset to their default values. */
void lbgpu::init() {

  LB_TRACE(printf("node %i Begin initialzing fluid on GPU\n", this_node));
  /** set parameters for transfer to gpu */
  lbgpu::reinit_parameters();

  if (lbdevicepar_gpu.number_of_particles)lbgpu::realloc_particles();
	
  lbgpu::init_GPU(&lbpar_gpu, &lbdevicepar_gpu);
#if 0
  if(this_node == 0)
  lb_lbfluid_save_checkpoint("checkpoint0.dat", 0);
  else
  lb_lbfluid_save_checkpoint("checkpoint1.dat", 0);
#endif
  LB_TRACE(printf("Initialzing fluid on GPU successful\n"));
}

/*@}*/

/***********************************************************************/
/** \name MPI stuff */
/***********************************************************************/
void lbgpu::get_values_multigpu(LB_values_gpu *host_values){
  
  mpi_recv_fluid_gpu(this_node, host_values);

}
void lbgpu::get_bounds_multigpu(unsigned *bound_array){
  
  mpi_recv_fluid_boundary_flags_gpu(this_node, bound_array);

}
/*************** REQ_GETPARTS ************/
/*************** REQ_GETPARTS ************/
/**
 * @params host_data struct storing all needed particle data (Output)
 *
 * */
static void mpi_get_particles_lb(LB_particle_gpu *host_data)
{
  int n_part;
  int g, pnode;
  Cell *cell;
  int c;
  MPI_Status status;

  int i;	
  int *sizes;
  sizes = (int*) malloc(sizeof(int)*n_nodes);

  n_part = cells_get_n_particles();

  /* first collect number of particles on each node */
  MPI_Gather(&n_part, 1, MPI_INT, sizes, 1, MPI_INT, 0, comm_cart);

  /* just check if the number of particles is correct */
  if(this_node > 0){
    /* call slave functions to provide the slave datas */
    mpi_get_particles_slave_lb();
  }
  else {
    /* master: fetch particle informations into 'result' */
    g = 0;
    for (pnode = 0; pnode < n_nodes; pnode++) {
      if (sizes[pnode] > 0) {
        if (pnode == 0) {
          for (c = 0; c < local_cells.n; c++) {
            Particle *part;
            int npart;	
            int dummy[3] = {0,0,0};
            double pos[3];
            cell = local_cells.cell[c];
            part = cell->part;
            npart = cell->n;
            for (i=0;i<npart;i++) {
              memcpy(pos, part[i].r.p, 3*sizeof(double));
              fold_position(pos, dummy);
              host_data[i+g].p[0] = (float)pos[0];
              host_data[i+g].p[1] = (float)pos[1];
              host_data[i+g].p[2] = (float)pos[2];
								
              host_data[i+g].v[0] = (float)part[i].m.v[0];
              host_data[i+g].v[1] = (float)part[i].m.v[1];
              host_data[i+g].v[2] = (float)part[i].m.v[2];
#ifdef LB_ELECTROHYDRODYNAMICS
              host_data[i+g].mu_E[0] = (float)part[i].p.mu_E[0];
              host_data[i+g].mu_E[1] = (float)part[i].p.mu_E[1];
              host_data[i+g].mu_E[2] = (float)part[i].p.mu_E[2];
#endif
            }  
            g += npart;
          }  
        }
        else {
          MPI_Recv(&host_data[g], sizes[pnode]*sizeof(LB_particle_gpu), MPI_BYTE, pnode, REQ_GETPARTS,
          comm_cart, &status);
          g += sizes[pnode];
        }
      }
    }
  }
  COMM_TRACE(fprintf(stderr, "%d: finished get\n", this_node));
  free(sizes);
}

static void mpi_get_particles_slave_lb(){
 
  int n_part;
  int g;
  LB_particle_gpu *host_data_sl;
  Cell *cell;
  int c, i;

  n_part = cells_get_n_particles();

  COMM_TRACE(fprintf(stderr, "%d: get_particles_slave, %d particles\n", this_node, n_part));

  if (n_part > 0) {
    /* get (unsorted) particle informations as an array of type 'particle' */
    /* then get the particle information */
    host_data_sl = (LB_particle_gpu*) malloc(n_part*sizeof(LB_particle_gpu));
    
    g = 0;
    for (c = 0; c < local_cells.n; c++) {
      Particle *part;
      int npart;
      int dummy[3] = {0,0,0};
      double pos[3];
      cell = local_cells.cell[c];
      part = cell->part;
      npart = cell->n;

      for (i=0;i<npart;i++) {
        memcpy(pos, part[i].r.p, 3*sizeof(double));
        fold_position(pos, dummy);	
			
        host_data_sl[i+g].p[0] = (float)pos[0];
        host_data_sl[i+g].p[1] = (float)pos[1];
        host_data_sl[i+g].p[2] = (float)pos[2];

        host_data_sl[i+g].v[0] = (float)part[i].m.v[0];
        host_data_sl[i+g].v[1] = (float)part[i].m.v[1];
        host_data_sl[i+g].v[2] = (float)part[i].m.v[2];
#ifdef LB_ELECTROHYDRODYNAMICS
        host_data_sl[i+g].mu_E[0] = (float)part[i].p.mu_E[0];
        host_data_sl[i+g].mu_E[1] = (float)part[i].p.mu_E[1];
        host_data_sl[i+g].mu_E[2] = (float)part[i].p.mu_E[2];
#endif
      }
      g+=npart;
    }
    /* and send it back to the master node */
    MPI_Send(host_data_sl, n_part*sizeof(LB_particle_gpu), MPI_BYTE, 0, REQ_GETPARTS, comm_cart);
    free(host_data_sl);
  }  
}

static void mpi_send_forces_lb(LB_particle_force_gpu *host_forces){
	
  int n_part;
  int g, pnode;
  Cell *cell;
  int c;
  int i;	
  int *sizes;
  sizes = (int*) malloc(sizeof(int)*n_nodes);
  n_part = cells_get_n_particles();
  /* first collect number of particles on each node */
  MPI_Gather(&n_part, 1, MPI_INT, sizes, 1, MPI_INT, 0, comm_cart);

  /* call slave functions to provide the slave datas */
  if(this_node > 0) {
    mpi_send_forces_slave_lb();
  }
  else{
  /* fetch particle informations into 'result' */
  g = 0;
    for (pnode = 0; pnode < n_nodes; pnode++) {
      if (sizes[pnode] > 0) {
        if (pnode == 0) {
          for (c = 0; c < local_cells.n; c++) {
            int npart;	
            cell = local_cells.cell[c];
            npart = cell->n;
            for (i=0;i<npart;i++) {
              cell->part[i].f.f[0] += (double)host_forces[i+g].f[0];
              cell->part[i].f.f[1] += (double)host_forces[i+g].f[1];
              cell->part[i].f.f[2] += (double)host_forces[i+g].f[2];
            }
 	    g += npart;
          }
        }
        else {
        /* and send it back to the slave node */
        MPI_Send(&host_forces[g], sizes[pnode]*sizeof(LB_particle_force_gpu), MPI_BYTE, pnode, REQ_GETPARTS, comm_cart);			
        g += sizes[pnode];
        }
      }
    }
  }
  COMM_TRACE(fprintf(stderr, "%d: finished send\n", this_node));

  free(sizes);
}

static void mpi_send_forces_slave_lb(){

  int n_part;
  LB_particle_force_gpu *host_forces_sl;
  Cell *cell;
  int c, i;
  MPI_Status status;

  n_part = cells_get_n_particles();

  COMM_TRACE(fprintf(stderr, "%d: send_particles_slave, %d particles\n", this_node, n_part));


  if (n_part > 0) {
    int g = 0;
    /* get (unsorted) particle informations as an array of type 'particle' */
    /* then get the particle information */
    host_forces_sl = (LB_particle_force_gpu*) malloc(n_part*sizeof(LB_particle_force_gpu));
    MPI_Recv(host_forces_sl, n_part*sizeof(LB_particle_force_gpu), MPI_BYTE, 0, REQ_GETPARTS,
    comm_cart, &status);
    for (c = 0; c < local_cells.n; c++) {
      int npart;	
      cell = local_cells.cell[c];
      npart = cell->n;
      for (i=0;i<npart;i++) {
        cell->part[i].f.f[0] += (double)host_forces_sl[i+g].f[0];
        cell->part[i].f.f[1] += (double)host_forces_sl[i+g].f[1];
        cell->part[i].f.f[2] += (double)host_forces_sl[i+g].f[2];
      }
      g += npart;
    }
    free(host_forces_sl);
  } 
}
/*@}*/

int lbgpu::lbnode_set_extforce_GPU(int ind[3], double f[3])
{
  if ( ind[0] < 0 || ind[0] >=  lbpar_gpu.dim_x ||
       ind[1] < 0 || ind[1] >= lbpar_gpu.dim_y ||
       ind[2] < 0 || ind[2] >= lbpar_gpu.dim_z )
    return ES_ERROR;

  unsigned int index =
    ind[0] + ind[1]*lbpar_gpu.dim_x + ind[2]*lbpar_gpu.dim_x*lbpar_gpu.dim_y;

  size_t  size_of_extforces = (n_extern_nodeforces+1)*sizeof(LB_extern_nodeforce_gpu);
  host_extern_nodeforces = (LB_extern_nodeforce_gpu*) realloc(host_extern_nodeforces, size_of_extforces);
  
  host_extern_nodeforces[n_extern_nodeforces].force[0] = (float)f[0];
  host_extern_nodeforces[n_extern_nodeforces].force[1] = (float)f[1];
  host_extern_nodeforces[n_extern_nodeforces].force[2] = (float)f[2];
  
  host_extern_nodeforces[n_extern_nodeforces].index = index;
  n_extern_nodeforces++;
  
  if(lbpar_gpu.external_force == 0)lbpar_gpu.external_force = 1;

  lbgpu::init_extern_nodeforces_GPU(n_extern_nodeforces, host_extern_nodeforces, &lbpar_gpu, &lbdevicepar_gpu);

  return ES_OK;
}

#endif /* LB_GPU */
