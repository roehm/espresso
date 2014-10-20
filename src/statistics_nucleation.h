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

int q6_ri_calculation();

int q6_calculation();

int q6_assign_ave();

void q6_pre_init();

int q6_initialize(double tcl_rc, double tcl_q6q6_min, int tcl_min_solid_bonds);

void q6_update();

int q6_average();

void q6_average_update();

void q6_assign_average();

void update_mean_part_pos();

void reset_mean_part_pos();

/** add q6 to another. This is used when collecting ghost q6. */
#ifdef Q6_PARA
MDINLINE void add_q6(ParticleQ6 *q6_to, ParticleQ6 *q6_add)
{
    int old_neb = q6_to->neb;
    q6_to->neb += q6_add->neb;
    for(int i=old_neb; i<q6_to->neb; i++){
      q6_to->neighbors[i] = q6_add->neighbors[i-old_neb];
    }
    //q6_to->solid_bonds += q6_add->solid_bonds;
    //fprintf(stderr, "ghostadd q6 solid bounds_to %i solid bounds_add %i \n", q6_to->solid_bonds, q6_add->solid_bonds);
    for (int m=0; m<=6; m++){
      //fprintf(stderr,"neb %i q6r=%f q6i=%f \n ",q6_add->neb,q6_add->q6r[m],q6_add->q6i[m]);
	     q6_to->q6r[m] += q6_add->q6r[m];
	     q6_to->q6i[m] += q6_add->q6i[m];
    }    
      //fprintf(stderr, "ghostadd q6 %lf neb %i \n", q6_to->q6, q6_to->neb);
}

MDINLINE void add_q6_solid_bonds(ParticleQ6 *q6_to, ParticleQ6 *q6_add)
{
    q6_to->solid_bonds += q6_add->solid_bonds;
    //fprintf(stderr, "ghostadd q6 solid bounds_to %i solid bounds_add %i \n", q6_to->solid_bonds, q6_add->solid_bonds);
}

MDINLINE double pair_q6q6( Particle *p1, Particle *p2 ) {

    double q6q6;

    //fprintf(stderr,"Check %f %f\n",p1->q.q6,p2->q.q6);

    q6q6  = ( p1->q.q6r[0] * p2->q.q6r[0] + p1->q.q6i[0] * p2->q.q6i[0] );
    for (int m=1; m<=6; m++){
	      q6q6 +=2*( p1->q.q6r[m] * p2->q.q6r[m] + p1->q.q6i[m] * p2->q.q6i[m]);
    }
    q6q6 /= ( p1->q.q6 * p2->q.q6 ); //why normalise by these factors? Tanja?
    q6q6 *= (4.0 * M_PI) / 13.0; //normalise by 4pi/13

    return( q6q6 );
}
#endif
#endif
