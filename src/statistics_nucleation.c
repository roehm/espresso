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
/** \file statistics_nucleation.c
 *
 *  This file contains the largest cluster of neighbor particles and the q6 algorithm.
 *
 */

#include <math.h>
#include "statistics.h"
#include "domain_decomposition.h"
#include "statistics_nucleation.h"
#include "communication.h"
#include "cells.h"
#include "particle_data.h"
#include "global.h"
#include "initialize.h"


#ifdef Q6_PARA 
#define MEAN_OFF
/*
##################################################################################################
 Begin q6:
*/
/** init of the q6 parameter, set via tcl */
Q6_Parameters q6para = {0.0, 0.0, 0.0};

/** calculation of spherical harmonics for l=6
 * @param *p_tmp		pointer to paricle struct
 * @param dr	distance (length) between paricles 
 * @param dx, dy, dz x,y,z distance vector entries
*/
void y6(Particle *p_tmp, double dr, double dx, double dy, double dz){

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
    }
}

/** calculates part[i].q.q6r[m] and part[i].q.q6i[m] of every particle
 * 
*/
int q6_ri_calculation(){

    on_observable_calc();
    double rc = q6para.rc;
    Particle *p1, *p2, **pairs;
    int c, i, m, n, np;
    double dist2;
    double rclocal2 = rc*rc; // sphere radius squared around particle for neighbor detection
    double vec21[3];
    int statusOK = 1;
    //int dummy[3] = {0,0,0};
    //totneb = 0;
//fprintf(stderr, "%d: rclocal2 %lf \n", this_node, rclocal2); 
    //int n_part;
    //int g, pnode;
    Cell *cell;
    Particle *part;
    //MPI_Status status;
    
    //part on node
    //n_part = cells_get_n_particles();
    
    for (int c = 0; c < local_cells.n; c++) {
      part = local_cells.cell[c]->part;
      np = local_cells.cell[c]->n;
       
      for (i=0;i<np;i++) {
        part[i].q.neb = 0;
        part[i].q.solid_bonds=0;
        //part[i].l.solid = 0;
        part[i].q.q6=0.0;
	       for (int m=0; m<=6; m++){
	         part[i].q.q6r[m]=0.0;
	         part[i].q.q6i[m]=0.0;
	       }
      }
    }
    for(c=0; c<ghost_cells.n; c++) {
      part = ghost_cells.cell[c]->part;
      np   = ghost_cells.cell[c]->n;
      for (i=0;i<np;i++) {
        part[i].q.neb = 0;
        part[i].q.solid_bonds=0;
        //part[i].l.solid = 0;
        part[i].q.q6=0.0;
	       for (m=0; m<=6; m++){
	         part[i].q.q6r[m]=0.0;
	         part[i].q.q6i[m]=0.0;
	       }

      }
    }    
    
      
    /* Loop local cells */
    for (c = 0; c < local_cells.n; c++) {

      VERLET_TRACE(fprintf(stderr,"%d: cell %d with %d neighbors\n",this_node,c, dd.cell_inter[c].n_neighbors));
      /* Loop cell neighbors */
      for (n = 0; n < dd.cell_inter[c].n_neighbors; n++) {
        pairs = dd.cell_inter[c].nList[n].vList.pair;
        np    = dd.cell_inter[c].nList[n].vList.n;
        VERLET_TRACE(fprintf(stderr,"%d: neighbor %d has %d particles\n",this_node,n,np));

        /* verlet list loop */
        for(i=0; i<2*np; i+=2) {
	         p1 = pairs[i];                    /* pointer to particle 1 */
	         p2 = pairs[i+1];                  /* pointer to particle 2 */
	         #ifdef MEAN_OFF
          dist2 = distance2vec(p2->r.p, p1->r.p, vec21);
          //fprintf(stderr, "%i: dist2 %lf vec %lf %lf %lf\n", p1->p.identity, dist2, vec21[0], vec21[1], vec21[2]);
          #else
	         //fold_position(p1->l.mean_pos, dummy);
          //fold_position(p2->l.mean_pos, dummy);
          //dist2 = distance2vec(p2->l.mean_pos, p1->l.mean_pos, vec21);
          #endif
	         if(dist2 < rclocal2) {
            #if 1
            if((p1->q.neb >= 27 || p2->q.neb >= 27)) {
              fprintf(stderr,"ERROR: Particle has more neighbors than possible! p1: %i p2: %i ", p1->q.neb, p2->q.neb);              
              errexit();
            }
            #endif
            p1->q.neighbors[p1->q.neb]=p2->p.identity;
            p2->q.neighbors[p2->q.neb]=p1->p.identity;
            p1->q.neb++;
            p2->q.neb++;
            if(dist2 != 0.0) y6(p1, sqrt(dist2), vec21[0], vec21[1], vec21[2]);
            if(dist2 != 0.0) y6(p2, sqrt(dist2), vec21[0], vec21[1], vec21[2]);
          }
        }
      }
    }
    ghost_communicator(&cell_structure.collect_ghost_q6_comm);
#if 0    
    for(c=0; c<ghost_cells.n; c++) {
      part = ghost_cells.cell[c]->part;
      np   = ghost_cells.cell[c]->n;
      for (i=0;i<np;i++) {
	       // Wolfgang Lechner and Christoph Dellago 2008 eq(1)
        if(part[i].q.neb > 0) {
          for (m=0; m<=6; m++){
            part[i].q.q6r[m] /= (float) part[i].q.neb;
            part[i].q.q6i[m] /= (float) part[i].q.neb;
            fprintf(stderr,"ghost particle %d Q6r %f Q6i: %f\n",part[i].p.identity,part[i].q.q6r[m],part[i].q.q6i[m]);
          }
        } else {
	           //Q6 undefined... system needs to collapse a little
	           //Q6 = 0.0;
	           for (m=0; m<=6; m++){
	             part[i].q.q6r[m] = 0.0;
	             part[i].q.q6i[m] = 0.0;
	           }  
	           statusOK = 0;
	         }
#if 1
	       part[i].q.q6 = 0.5 * ( part[i].q.q6r[0] * part[i].q.q6r[0] + part[i].q.q6i[0] * part[i].q.q6i[0] );
	       //fprintf(stderr, "Anfang: %f aus %f und %f\n", part[i].l.q6, part[i].q.q6r[0], part[i].q.q6i[0]);
	       for (int m=1; m<=6; m++){
	         part[i].q.q6 += part[i].q.q6r[m] * part[i].q.q6r[m] + part[i].q.q6i[m] * part[i].q.q6i[m];
	       }
	       //fprintf(stderr, "Ende: %f\n", part[i].l.q6);
	       part[i].q.q6 *= (4.0 * M_PI) / 13.0; //normalise by 4pi/13
        // Steinhardt order parameter: Wolfgang Lechner and Christoph Dellago 2008 eq(3)
	       part[i].q.q6 = sqrt(part[i].q.q6);    // This is the local invariant q6 per particle (Eq. 7 in ten Wolde)
        //fprintf(stderr,"Particle %d has %d neighbors. Q6: %f\n",part[i].p.identity,part[i].q.neb,part[i].q.q6);
        // Neigbor count optional
	       //totneb += part[i].q.neb;
#endif
      }      
    }    
#endif 

    for (c = 0; c < local_cells.n; c++) {
      cell = local_cells.cell[c];
      part = cell->part;
      np = cell->n;
       
      for (i=0;i<np;i++) {
	       // Wolfgang Lechner and Christoph Dellago 2008 eq(1)
        if(part[i].q.neb > 0) {
          for (m=0; m<=6; m++){
            part[i].q.q6r[m] /= (double) part[i].q.neb;
            part[i].q.q6i[m] /= (double) part[i].q.neb;
            //fprintf(stderr, "%i: real: %lf im: %lf \n", part[i].p.identity, part[i].q.q6r[m],part[i].q.q6i[m]);
            //fprintf(stderr,"Particle %d Q6r %f Q6: %f\n",part[i].p.identity,part[i].q.q6r[m],part[i].l.q6);
          }
        } else {
	           //Q6 undefined... 
	           for (m=0; m<=6; m++){
	             part[i].q.q6r[m] = 0.0;
	             part[i].q.q6i[m] = 0.0;
	             //fprintf(stderr, "hab kenen %i: %lf real %lf im \n", part[i].p.identity, part[i].q.q6r[m],part[i].q.q6i[m]);
	           }
	           //statusOK = 0;
	         }
	     } 
	   } 
    return statusOK;
}
/** calculates the local invariant q6 (part[i].q.q6) per particle
 * 
*/
int q6_calculation(){
    int c, i, m, np;
    Cell *cell;
    Particle *part;
    int statusOK = 1;
        
    for (c = 0; c < local_cells.n; c++) {
      cell = local_cells.cell[c];
      part = cell->part;
      np = cell->n;
       
      for (i=0;i<np;i++) {
	       part[i].q.q6 = 0.5 * ( part[i].q.q6r[0] * part[i].q.q6r[0] + part[i].q.q6i[0] * part[i].q.q6i[0] );
	       //fprintf(stderr, "Anfang: %f aus %f und %f\n", part[i].l.q6, part[i].q.q6r[0], part[i].q.q6i[0]);
	       for (m=1; m<=6; m++){
	         part[i].q.q6 += part[i].q.q6r[m] * part[i].q.q6r[m] + part[i].q.q6i[m] * part[i].q.q6i[m];
	       }
	       //fprintf(stderr, "Ende: %f\n", part[i].q.q6);
	       part[i].q.q6 *= (4.0 * M_PI) / 13.0; //normalise by 4pi/13
        // Steinhardt order parameter: Wolfgang Lechner and Christoph Dellago 2008 eq(3)
	       part[i].q.q6 = sqrt(part[i].q.q6);    // This is the local invariant q6 per particle (Eq. 7 in ten Wolde)
	       //if(part[i].q.q6_mean == 0.0) part[i].q.q6_mean = part[i].q.q6;
	       //part[i].q.q6_mean = (part[i].q.q6 + part[i].q.q6_mean)/2;
        //fprintf(stderr,"Particle %d has %d neighbors. Q6: %f\n",part[i].p.identity,part[i].q.neb,part[i].q.q6);
        // Neigbor count optional
	       //totneb += part[i].q.neb;
      }      
    }    

    return statusOK;
}

/** calculates the average of local invariant q6 (part[i].q.q6) per particle
    using the method of Wolfgang Lechner and Christoph Dellago 2008
 * 
*/

void q6_average(){

    //TODO: need? on_observable_calc();
    double Q6r[7], Q6i[7]; //global Q6m, real and imaginary part
    int c, i, m, k;
    int np;
    //int g, pnode;
    //Cell *cell;
    Particle *part, *part2;

    //MPI_Status status;
    
    //part on node
    //n_part = cells_get_n_particles();
    
    for (c = 0; c < local_cells.n; c++) {
      part = local_cells.cell[c]->part;
      np = local_cells.cell[c]->n;
       
      for (i=0;i<np;i++) {
        part[i].q.q6_ave=0.0;
        //for (int m=0; m<=6; m++){
        //fprintf(stderr,"real particle %d q6i vor com: %f\n",part[i].p.identity,part[i].q.q6r[m]);
        //}
      }
    }
    for(c=0; c<ghost_cells.n; c++) {
      part = ghost_cells.cell[c]->part;
      np   = ghost_cells.cell[c]->n;
      
      for (i=0;i<np;i++) {
        part[i].q.q6_ave=0.0;
        //for (int m=0; m<=6; m++){
        //fprintf(stderr,"ghost particle %d q6i vor com: %f\n",part[i].p.identity,part[i].q.q6r[m]);
        //}
      }
    }

   
    ghost_communicator(&cell_structure.update_ghost_q6_comm);

#if 0
    for(c=0; c<ghost_cells.n; c++) {
      part = ghost_cells.cell[c]->part;
      np   = ghost_cells.cell[c]->n;
      for (i=0;i<np;i++) {
	       fprintf(stderr,"ghost particle %d q6i nach com: %f\n",part[i].p.identity,part[i].q.q6r);
      }
    }       
#endif 

    for (int c = 0; c < local_cells.n; c++) {
      part = local_cells.cell[c]->part;
      np = local_cells.cell[c]->n;
       
      for (i=0;i<np;i++) {
	       // Wolfgang Lechner and Christoph Dellago 2008 eq(1)
        if(part[i].q.neb > 0) {
        // init
          for (m=0; m<=6; m++){
	           Q6r[m] = 0.0;
	           Q6i[m] = 0.0;
          }         
	         for(k=0; k<part[i].q.neb; k++){
	      
	           part2 = local_particles[part[i].q.neighbors[k]];
	             for (int m=0; m<=6; m++){
	                 //neighbor particle q6r, q6i
	               Q6r[m] += part2->q.q6r[m];
	               Q6i[m] += part2->q.q6i[m];
	                 //fprintf(stderr,"particle %d neb Q6r: %f\n",part[i].p.identity,Q6r[m]);
	             }
	           
	         }
	       	   //add values of particle itself and
	           //divide with number of neighbors + particle itself (lechner and dellago 2008 eq(6))
	         for (int m=0; m<=6; m++){
	           Q6r[m] = (Q6r[m] + part[i].q.q6r[m])/(double)(part[i].q.neb + 1);
	           Q6i[m] = (Q6i[m] + part[i].q.q6i[m])/(double)(part[i].q.neb + 1);
	         }	   
	       } else {
		        //Q6 is undefined
		        for (int m=0; m<=6; m++){
		          Q6r[m] = 0.0;
            Q6i[m] = 0.0;
          }        
	       } 
      // calc average q6 lechner and dellago 2008 eq(5)
      part[i].q.q6_ave = 0.5 * ( Q6r[0]*Q6r[0] + Q6i[0]*Q6i[0] );
      for (int m=1; m<=6; m++){
	      part[i].q.q6_ave += Q6r[m]*Q6r[m] + Q6i[m]*Q6i[m];
      }
      part[i].q.q6_ave *= 4.0 * M_PI/13.0;
      part[i].q.q6_ave = sqrt(part[i].q.q6_ave);
      //if(part[i].q.q6_ave>10.0)fprintf(stderr,"particle %d ave_q6: %f\n",part[i].p.identity,part[i].q.q6_ave);
	       
	    }
	  } 
}

inline double pair_q6q6( Particle *p1, Particle *p2 ) {

    double q6q6;

    //fprintf(stderr,"Check %f %f\n",p1->q.q6,p2->q.q6);

    q6q6  = 0.5 * ( p1->q.q6r[0] * p2->q.q6r[0] + p1->q.q6i[0] * p2->q.q6i[0] );
    for (int m=1; m<=6; m++){
	    q6q6 += p1->q.q6r[m] * p2->q.q6r[m] + p1->q.q6i[m] * p2->q.q6i[m];
    }
    q6q6 /= ( p1->q.q6 * p2->q.q6 ); //why normalise by these factors? Tanja?
    q6q6 *= (4.0 * M_PI) / 13.0; //normalise by 4pi/13

    return( q6q6 );
}

int q6_assign_ave(){


//TODO: is
    on_observable_calc(); 
    //necessary?
    //solidParticles = 0;
    //bondCount      = 0;
    int np;
    Particle *part1, *part2, **pairs;
    int i, n, c;

    
    for (c = 0; c < local_cells.n; c++) {
      part1 = local_cells.cell[c]->part;
      np = local_cells.cell[c]->n;
       
      for (i=0;i<np;i++) {
        part1[i].q.solid_bonds=0;
        part1[i].q.solid_state=0;
      }
    }
    for(c=0; c<ghost_cells.n; c++) {
      part1 = ghost_cells.cell[c]->part;
      np   = ghost_cells.cell[c]->n;
      
      for (i=0;i<np;i++) {
        part1[i].q.solid_bonds=0;
        part1[i].q.solid_state=0;
      }
    }

    ghost_communicator(&cell_structure.update_ghost_q6_comm);
    /* Loop local cells */
    for (c = 0; c < local_cells.n; c++) {

      VERLET_TRACE(fprintf(stderr,"%d: cell %d with %d neighbors\n",this_node,c, dd.cell_inter[c].n_neighbors));
      /* Loop cell neighbors */
      for (n = 0; n < dd.cell_inter[c].n_neighbors; n++) {
        pairs = dd.cell_inter[c].nList[n].vList.pair;
        np    = dd.cell_inter[c].nList[n].vList.n;
        VERLET_TRACE(fprintf(stderr,"%d: neighbor %d has %d particles\n",this_node,n,np));

        /* verlet list loop */
        for(i=0; i<2*np; i+=2) {
	         part1 = pairs[i];
	         part2 = pairs[i+1];
           if(part1->q.q6 != 0.0 && part2->q.q6 != 0.0){
               part1->q.q6q6 = pair_q6q6(part1, part2);
               part2->q.q6 = part1->q.q6;
           } else {
               part1->q.q6q6 = 0.0;
               part2->q.q6q6 = 0.0;
             }
           //TODO find out if negative q6q6 is correct
           //if(part1->q.q6q6 <0.0) printf("partcle %i q6q6: %f bonds: %i\n", part1->p.identity,part1->q.q6q6, part1->q.solid_bonds);

           //Test against arbitrary threshold
           if(part1->q.q6q6 > q6para.q6q6_min) {
              part1->q.solid_bonds++;
              part2->q.solid_bonds++;
           }

           //fprintf(stderr,"solid_bonds: %i\n", part1->q.solid_bonds);
           //accumulate an average stat for the whole system
           //if( p1->l.neighbors[j] > i ) { //avoid double-counting
	         //  eQ6Q6 += p1->l.q6q6;
	         //  bondCount++;
           //}
        }//vv loop 
      }// neighbor loop
          
    }//cells loops
    //communicate the number of bonds from ghosts to real part 
    ghost_communicator(&cell_structure.collect_ghost_q6_comm);

    for (c = 0; c < local_cells.n; c++) {
      part1 = local_cells.cell[c]->part;
      np = local_cells.cell[c]->n;
       
      for (i=0;i<np;i++) {
        if(part1[i].q.solid_bonds >= q6para.min_solid_bonds) {
	         part1[i].q.solid_state = 1;
        }
      }
    }
    //reduce to get the average
    //if( bondCount > 0 ) {
    //    eQ6Q6 /= (double) bondCount;
    //}

  //fprintf(stderr,"solidParticles %d:\n",solidParticles);
  return 1;
}
/** initializes and communicates the tcl parameters for q6 usage
 
*/
int q6_initialize(double tcl_rc, double tcl_q6q6_min, int tcl_min_solid_bonds) {

    q6para.rc = tcl_rc;
    q6para.q6q6_min = tcl_q6q6_min;
    q6para.min_solid_bonds = tcl_min_solid_bonds;
    mpi_bcast_q6_params();
    reset_mean_part_pos();
    //printf("bcast q6 params ok\n");
    
    return 0;

}
/** updates (recalcs) the initializes local invariant q6 with mpi
 * 
*/
void q6_update() {

    mpi_q6_calculation();

    //printf("mpi_q6_calculation update ok\n");

}

void q6_average_update() {

    mpi_q6_average_calculation(); 

}

void q6_assign_average() {

    mpi_q6_assign_average_calculation();

}
/**********************************************************************************/
/** pre_init (for later use or can be removed)
 * 
*/
void q6_pre_init() {
    
    reset_mean_part_pos();
    
}

//only 4 my personal use


/** updates the mean calc of q6 per particle
 * 
*/

void update_mean_q6_calculation(){


}

void update_mean_part_pos(){
    
    int np;
    Cell *cell;
    int c, i;
    Particle *part;
    
    /* Loop local cells */
    for (c = 0; c < local_cells.n; c++) {
      cell = local_cells.cell[c];
      part = cell->part;
      np  = cell->n;
#if 0      
      for (i=0;i<np;i++) {
	       part[i].l.mean_pos[0] = (part[i].r.p[0]+part[i].l.mean_pos[0])/2;
	       part[i].l.mean_pos[1] = (part[i].r.p[1]+part[i].l.mean_pos[1])/2;
	       part[i].l.mean_pos[2] = (part[i].r.p[2]+part[i].l.mean_pos[2])/2;
	     }
#endif	     
	   }
//printf("udate mean pos finished\n");
}

void reset_mean_part_pos(){

    int np;
    Cell *cell;
    int c, i;
    Particle *part;

    /* Loop local cells */
    for (c = 0; c < local_cells.n; c++) {
      cell = local_cells.cell[c];
      part = cell->part;
      np  = cell->n;
    
      for (i=0;i<np;i++) {
        part[i].q.q6_mean = 0.0;
#if 0
	       part[i].l.mean_pos[0] = part[i].r.p[0];
	       part[i].l.mean_pos[1] = part[i].r.p[1];
	       part[i].l.mean_pos[2] = part[i].r.p[2];
#endif
	     }	    

	   }

}
#endif
