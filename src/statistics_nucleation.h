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

//spherical harmoic for l=6
//#if 0
MDINLINE void y6(Particle *p_tmp, double dr, double dx, double dy, double dz){

    double costh,costh2,costh4,costh6,sinth,sinth2,sinth3,sinth4,sinth5,sinth6;
    double cosphi1,cosphi2,cosphi3,cosphi4,cosphi5,cosphi6,twocosphi;
    double sinphi1,sinphi2,sinphi3,sinphi4,sinphi5,sinphi6;
    double Ba;
    double y6[2][7]; // spher. harm. real and imaginary part

    // static const ?
    double y6c0 = 1.0/32.0 * sqrt(13.0/M_PI);
    double y6c1 = 1.0/16.0 * sqrt(273.0/(2*M_PI));
    double y6c2 = 1.0/64.0 * sqrt(1365.0/M_PI);
    double y6c3 = 1.0/32.0 * sqrt(1365.0/M_PI);
    double y6c4 = 3.0/32.0 * sqrt(91.0/(2.0*M_PI));
    double y6c5 = 3.0/32.0 * sqrt(1001.0/M_PI);
    double y6c6 = 1.0/64.0 * sqrt(3003.0/M_PI);

    costh = dz / dr;
    costh2 = costh*costh;
    costh4 = costh2*costh2;
    costh6 = costh2*costh4;
    sinth2 = 1.0 - costh2;
    sinth = sqrt(sinth2);
    sinth3 = sinth*sinth2;
    sinth4 = sinth2*sinth2;
    sinth5 = sinth2*sinth3;
    sinth6 = sinth3*sinth3;

    if(sinth < 0.000001) {
	    cosphi1 = 0.0;
	    sinphi1 = 0.0;
    } else {
	    cosphi1 = dx/(dr*sinth);
	    sinphi1 = dy/(dr*sinth);
    }

    twocosphi = 2.0*cosphi1;
    cosphi2 = twocosphi*cosphi1 - 1.0;
    sinphi2 = twocosphi*sinphi1;
    cosphi3 = twocosphi*cosphi2 - cosphi1;
    sinphi3 = twocosphi*sinphi2 - sinphi1;
    cosphi4 = twocosphi*cosphi3 - cosphi2;
    sinphi4 = twocosphi*sinphi3 - sinphi2;
    cosphi5 = twocosphi*cosphi4 - cosphi3;
    sinphi5 = twocosphi*sinphi4 - sinphi3;
    cosphi6 = twocosphi*cosphi5 - cosphi4;
    sinphi6 = twocosphi*sinphi5 - sinphi4;

    y6[0][0] = y6c0*(-5.0+105.0*costh2-315.0*costh4+231.0*costh6 );
    y6[1][0] = 0.0;
    Ba = y6c1 * costh*(5.0 - 30.0 * costh2 + 33.0*costh4)*sinth;
    y6[0][1] = - Ba * cosphi1;
    y6[1][1] = - Ba * sinphi1;
    Ba = y6c2 * (1.0 - 18.0 * costh2+ 33.0 * costh4)* sinth2;
    y6[0][2]= Ba *  cosphi2;
    y6[1][2]= Ba * sinphi2;
    Ba = y6c3 * costh*(-3.0 + 11.0*costh2)*sinth3;
    y6[0][3]= - Ba *  cosphi3;
    y6[1][3]= - Ba * sinphi3;
    Ba =  y6c4*(-1.0 + 11.0 * costh2)* sinth4;
    y6[0][4]= Ba *  cosphi4;
    y6[1][4]= Ba * sinphi4;
    Ba = y6c5 * costh * sinth5;
    y6[0][5]= - Ba *  cosphi5;
    y6[1][5]= - Ba * sinphi5;
    Ba = y6c6* sinth6;
    y6[0][6]= Ba *  cosphi6;
    y6[1][6]= Ba * sinphi6;

    //fprintf(stderr,"ID: %d, adding ", p_tmp->p.identity, dr, dx, dy, dz);

    for (int m=0; m<=6; m++){
        //fprintf(stderr,"q6r=%f q6i=%f, ",y6[0][m],y6[1][m]);
	    p_tmp->q.q6r[m] += y6[0][m];
	    p_tmp->q.q6i[m] += y6[1][m];
	    //fprintf(stderr, "%i: neb %lf real \n", p_tmp->p.identity, p_tmp->q.q6r[m]);
     //fprintf(stderr, "part %i: %lf im \n",p_tmp->p.identity, p_tmp->q.q6i[m]);
    }
    
    //fprintf(stderr,"\n");
}

/** add q6 to another. This is used when collecting ghost q6. */
MDINLINE void add_q6(ParticleQ6 *q6_to, ParticleQ6 *q6_add)
{
    q6_to->neb += q6_add->neb; 
    for (int m=0; m<=6; m++){
      //fprintf(stderr,"neb %i q6r=%f q6i=%f \n ",q6_add->neb,q6_add->q6r[m],q6_add->q6i[m]);
	     q6_to->q6r[m] += q6_add->q6r[m];
	     q6_to->q6i[m] += q6_add->q6i[m];
    }    
      //fprintf(stderr, "ghostadd q6 %lf neb %i \n", q6_to->q6, q6_to->neb);
}
//#endif
#endif
