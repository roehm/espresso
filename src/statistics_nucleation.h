/*
  Copyright (C) 2010 The ESPResSo project
  Copyright (C) 2002,2003,2004,2005,2006,2007,2008,2009,2010 Max-Planck-Institute for Polymer Research, Theory Group, PO Box 3148, 55021 Mainz, Germany
  
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
#ifndef STATISTICS_NUCLEATION_H
#define STATISTICS_NUCLEATION_H
/** \file statistics_nucleation.h
 *
 *  This file contains algorithms useful for nucleation.
 *  It can be used to identify the largest cluster of neighbor particles and
 *  for detecting whether particles are solid-like or not. Therefore, the Q6 algorithm is provided.
 *  
 *
 */

#include <tcl.h>

//struct for Q6 global vars
typedef struct {

  double rc;
  double q6q6_min;
  int min_solid_bonds;
  
} Q6_Parameters;

extern Q6_Parameters q6para;

int ql6_calculation();
int q6_calculation();

void q6_pre_init();

double analyze_bubble_volume(Tcl_Interp *interp, double bubble_cut, double sigma);

double analyze_q6(double tcl_rc, double tcl_q6q6_min, int tcl_min_solid_bonds);

double analyze_q6_solid(double tcl_rc, double tcl_q6q6_min, int tcl_min_solid_bonds);

double analyze_q6_solid_cluster(double tcl_rc, double tcl_q6q6_min, int tcl_min_solid_bonds);

int initialize_q6(double tcl_rc, double tcl_q6q6_min, int tcl_min_solid_bonds);

double reduceQ6Q6();

void update_q6();

void update_mean_part_pos();

void reset_mean_part_pos();

/** add q6 to another. This is used when collecting ghost q6. */
MDINLINE void add_q6(ParticleQ6 *q6_to, ParticleQ6 *q6_add)
{
    int old_neb = q6_to->neb;
    q6_to->neb += q6_add->neb;
    for(int i=old_neb; i<q6_to->neb; i++){
      q6_to->neighbors[i] = q6_add->neighbors[i-old_neb];
    }
    for (int m=0; m<=6; m++){
      //fprintf(stderr,"neb %i q6r=%f q6i=%f \n ",q6_add->neb,q6_add->q6r[m],q6_add->q6i[m]);
	     q6_to->q6r[m] += q6_add->q6r[m];
	     q6_to->q6i[m] += q6_add->q6i[m];
    }    
      //fprintf(stderr, "ghostadd q6 %lf neb %i \n", q6_to->q6, q6_to->neb);
}
//#endif
#endif
