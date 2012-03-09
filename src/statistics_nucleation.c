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


#ifdef Q6_PARA 
/* convert the 1D array index to 3d cell coordinates */
void convert_index_to_coordinates(int *box_n, int *ncell, int the_index) {
        int tmp_index = the_index;

        ncell[2]=(int) tmp_index / (box_n[0]*box_n[1]);

        if(the_index > ncell[2]*box_n[0]*box_n[1]) {
                tmp_index = tmp_index - ncell[2]*box_n[0]*box_n[1];
                ncell[1]=(int) tmp_index / box_n[0];
        } else {
                ncell[1]=0;
        }

        if(tmp_index > ncell[1]*box_n[0]) {
                ncell[0]=(int) (tmp_index - ncell[1]*box_n[0]);
        } else {
                ncell[0]=0;
        }
}

/* convert the 3d coordinates to 1d array index */
int convert_coordinates_to_index(int *box_n, int *ncell) {
        int the_index = 0;
        the_index = ncell[0];
        if ((ncell[1] != 0) && (ncell[2] != 0)) {
                the_index = the_index + ncell[1]*box_n[0] + ncell[2]*box_n[0]*box_n[1];
        } else if (ncell[1] != 0) {
                the_index = the_index + ncell[1]*box_n[0];
        } else if (ncell[2] != 0) {
                the_index = the_index + ncell[2]*box_n[0]*box_n[1];
        }
        return the_index;
}

/* calculate index of neigbor cell in every direction with PBC */
int check_cell_neighbor(int *box_n, int act_index, int direction) {
        int max_index = box_n[0]*box_n[1]*box_n[2];
        int cell_index[3];
        cell_index[0]=0;
        cell_index[1]=0;
        cell_index[2]=0;
        int tmp_index = -1;
        convert_index_to_coordinates(box_n, cell_index, act_index);

        switch (direction) {
                case 0:
                        if(cell_index[0] < box_n[0]) {
                                cell_index[0]++;
                        } else {
                                cell_index[0] -= box_n[0];
                        }
                        tmp_index=convert_coordinates_to_index(box_n, cell_index);
                        if ((tmp_index >= 0) && (tmp_index <= max_index)) return tmp_index;
                        break;
                case 1:
                        if(cell_index[0] > 0) {
                                cell_index[0]--;
                        } else {
                                cell_index[0] += box_n[0];
                        }
                        tmp_index=convert_coordinates_to_index(box_n, cell_index);
                        if ((tmp_index >= 0) && (tmp_index <= max_index)) return tmp_index;
                        break;
                case 2:
                        if(cell_index[1] < box_n[1]) {
                                cell_index[1]++;
                        } else {
                                cell_index[1] -= box_n[1];
                        }
                        tmp_index=convert_coordinates_to_index(box_n, cell_index);
                        if ((tmp_index >= 0) && (tmp_index <= max_index)) return tmp_index;
                        break;
                case 3:
                        if(cell_index[1] > 0) {
                                cell_index[1]--;
                        } else {
                                cell_index[1] += box_n[1];
                        }
                        tmp_index=convert_coordinates_to_index(box_n, cell_index);
                        if ((tmp_index >= 0) && (tmp_index <= max_index)) return tmp_index;
                        break;
                case 4:
                        if(cell_index[2] < box_n[2]) {
                                cell_index[2]++;
                        } else {
                                cell_index[2] -= box_n[2];
                        }
                        tmp_index=convert_coordinates_to_index(box_n, cell_index);
                        if ((tmp_index >= 0) && (tmp_index <= max_index)) return tmp_index;
                        break;

                case 5:
                        if(cell_index[2] > 0) {
                                cell_index[2]--;
                        } else {
                                cell_index[2] += box_n[2];
                        }
                        tmp_index=convert_coordinates_to_index(box_n, cell_index);
                        if ((tmp_index >= 0) && (tmp_index <= max_index)) return tmp_index;
                        break;
                default:
                        return -1;

        }
        return -1;
}

/* visit cell, (partial) recursive function */
int visit_cell(int i, int v_cnt, int *vapliq, int *box_n, int *tmp_size) {

        int tmp_visit = -1;
        int dir_count = 0;
        /* loop all cells */
        /* Check if cell is already labelled */
        if (vapliq[i] == 1) {
                *tmp_size += 1;
                vapliq[i] = v_cnt;
                /* Analyze cluster and write higher values */
                /* 0: x; 1: -x, 2: y, 3: -y, 4: z, 5: -z */
                for (dir_count=0;dir_count<6; dir_count++) {
                        tmp_visit = check_cell_neighbor(box_n, i, dir_count);
                        if (tmp_visit != -1) visit_cell(tmp_visit,v_cnt,vapliq,box_n,tmp_size);
                }
        }
        return 0;
}

/* Particle identified as liquid-like. Label Cell with particle and cells around. */
void add_liquid_particle(double *p, int *vapliq, int *box_n, double sigma, double sigmah, long long int index_max) {

        int ncell[3];
        double xi,yi,zi;
        int vlindex = 0;

        int abortflag=0;
        double xii,yii,zii;

        /* Label cells around liquid particle as liquid-like */
        for(xi = -1.6*sigma; xi <= 1.6*sigma; xi+=sigmah) {
                for(yi = -1.6*sigma; yi <= 1.6*sigma; yi+=sigmah) {
                        for(zi = -1.6*sigma; zi <= 1.6*sigma; zi+=sigmah) {

                                abortflag=0;

                                ncell[0]=(int)((p[0]+xi)/sigmah);
                                ncell[1]=(int)((p[1]+yi)/sigmah);
                                ncell[2]=(int)((p[2]+zi)/sigmah);

                                for(xii=0;xii<=sigmah;xii+=sigmah) {
                                        for(yii=0;yii<=sigmah;yii+=sigmah) {
                                                for(zii=0;zii<=sigmah;zii+=sigmah) {
                                                        // necessary because of (int)-cast. Otherwise xi+xii would be enough
                                                        if(sqrt(pow((ncell[0]*sigmah+xii-p[0]),2.0)+pow((ncell[1]*sigmah+yii-p[1]),2.0)+pow((ncell[2]*sigmah+zii-p[2]),2.0)) > 1.6*sigma) abortflag=1;
                                                }
                                        }
                                }
                                if(!abortflag)
                                {
                                        if(ncell[0] < 0) {
                                                ncell[0] += box_n[0];
                                        }
                                        if(ncell[0] > box_n[0]) {
                                                ncell[0] -= box_n[0];
                                        }
                                        if(ncell[1] < 0) {
                                                ncell[1] += box_n[1];
                                        }
                                        if(ncell[1] > box_n[1]) {
                                                ncell[1] -= box_n[1];
                                        }
                                        if(ncell[2] < 0) {
                                                ncell[2] += box_n[2];
                                        }
                                        if(ncell[2] > box_n[2]) {
                                                ncell[2] -= box_n[2];
                                        }
                                        /*  labelling of box with box_n[0],box_n[1],box_n[2]   */
                                        if((ncell[0] >= 0) && (ncell[1] >= 0) && (ncell[2] >= 0)) {
                                                if((ncell[0] <= (box_n[0])) && (ncell[1] <= (box_n[1])) && (ncell[2] <= (box_n[2]))) {

                                                        vlindex = convert_coordinates_to_index(box_n, ncell);

                                                        if((vapliq[vlindex]!=0) && (vlindex < index_max)) {
                                                                // set to liquid
                                                                vapliq[vlindex]=0;
                                                        }
                                                }
                                        }

                                } // abortflag
                        }  // zi
                } // yi
        } // xi
}

/* init array for vapor liquid distinction */
void bubble_volume_init_vapliq(int *vapliq, long long int cell_cnt) {
        int i = 0;
        for(i = 0; i < cell_cnt; i++) {
                // set all to vapor ( >= 1 => vapor, 0 => liquid)
                vapliq[i] = 1;
        }
}

/* count neighbors of particle p */
int count_neighbors(double p_tmp[3], double radius) {
    IntList il;
    int planedims[3];
    int neb;
    int i;
    planedims[0] = planedims[1] = planedims[2] = 1;

    updatePartCfg(WITHOUT_BONDS);

    nbhood(p_tmp, radius, &il, planedims );
    //fprintf(stderr,"%d %f %f %f\n", il.n,p_tmp[j].r.p[0],p_tmp[j].r.p[1],p_tmp[j].r.p[2]);
    fprintf(stderr," Check: %d, Particle(s) ", il.n-1);
    for(i=0;i<il.n;i++)
        fprintf(stderr,"%d ",il.e[i]);
    fprintf(stderr,"\n");
    neb=il.n-1;
    //realloc_intlist(&il, 0);
    return neb;
}

double analyze_bubble_volume(Tcl_Interp *interp, double bubble_cut, double sigma) {

        double sigmah = sigma / 2.0;

        int c=0, np1=0, i=0 ,j=0;
        Cell *cell;
        Particle *p1;

        int init_vapliq = 0;

        int v_cnt = 2;                // counter for labelling vapor-like cells in cluster
        int biggest_cluster = 0;
        int box_n[3];
        int tmp_size = 0;
        long long int cell_cnt = 0;
        int biggest_cluster_label;

        /* Number of cells in every direction */
        for(i = 0; i < 3; i++) {
                box_n[i] = (int) (box_l[i] / sigmah) + 1;
        }

        cell_cnt = box_n[0] * box_n[1] * box_n[2];
        if(cell_cnt < 0) {
                Tcl_AppendResult(interp, "Error: Number of cells is negative!", (char *)NULL);
                return TCL_ERROR;
        }

        // TODO
        //int *vapliq = (int *)malloc(cell_cnt*sizeof(int));
        int vapliq[cell_cnt];
        for(i = 0; i < cell_cnt; i++) {
                // set all to vapor ( >= 1 => vapor, 0 => liquid)
                vapliq[i] = 1;
        }
        init_vapliq = 1;

       /* LIQUID PARTICLE SEARCH ALGORITHM */

        /* Loop over cells and neighbors, find particles with more than five neighbors */
        for (c = 0; c < local_cells.n; c++) {
            cell = local_cells.cell[c];
            p1   = cell->part;
            np1  = cell->n;

            for(j=0; j < np1; j++) {

              if(count_neighbors(p1[j].r.p,bubble_cut) > 5) {
                  add_liquid_particle(p1[j].r.p,vapliq,box_n,sigma, sigmah,cell_cnt);
              } // if > 5
            }  // np1: particles in local cell

        } // local_cells

        /* ADDED LIQUID PARTICLES */

       // fclose(fpncl);
       // fclose(fpncv);

        /* Analyze connected vapor like cells and calculate volume */
        tmp_size = 0;
        biggest_cluster_label = 0;
        if (init_vapliq==1) {
                for(i = 0; i < cell_cnt; i++) {
                        visit_cell(i,v_cnt,vapliq,box_n,&tmp_size);
                        if (biggest_cluster < tmp_size) {
                                biggest_cluster = tmp_size;
                                biggest_cluster_label = v_cnt;
                        }
                        tmp_size = 0;
                        v_cnt++;
                }
//                fprintf(stderr,"finding clusters done. Biggest is %d cells.\n",biggest_cluster);
        } else {
                // no neighbors were found, all vapor
                biggest_cluster = cell_cnt;
//                fprintf(stderr,"finding clusters done. All of the %d cells are vapor.\n",biggest_cluster);
        }

  return pow(0.5,3.0)*pow(sigma,3.0)*biggest_cluster;
}

/*
 ^ bubble volume done.
##################################################################################################

 Begin cluster algorithms
*/

// global variables
int haveClusters = 0;
int clusterCount;
int solidParticles = 0;
int totneb = 0;
Q6_Parameters q6para;
double    q6q6, eQ6Q6;
int       bondCount;

typedef struct particleCluster {
    int           size;
    int           rootMember;
    struct particleCluster  *next;

} particleCluster;

//function to start a new cluster
void init( particleCluster *curCluster, int root) {
      curCluster->size = 0;
      curCluster->next = 0;
      curCluster->rootMember = root;
}

struct particleCluster *clust;

//build clusters using a breadth-first graph algorithm
//Algorithm is on page 594 of Cormen, Leiserson & Rivest (3rd edition).
// Colors: Black = -2, Grey = -1
void buildClusters(int rootmember) {

    struct particleCluster *c;
    int           i, i_start, i_end, j, *clustFlags, *ParticleListP, p, rootId, nebId;
    int           clusterQueueHead, clusterQueueTail;
    Particle  *p1, *p2;
    ParticleListP = (int*) malloc(n_total_particles * sizeof(int));

    //clusterCount tracks number of clusters actually allocated
    clusterCount = 0;
    clustFlags   = (int*) malloc(n_total_particles * sizeof(int));

    // clean up
    while( clust != 0 ){
      c     = clust;
      clust = c->next;
      free(c);
    }

    //initialize the clustering
    for (i = 0; i < n_total_particles; i++){
        ParticleListP[i]    = -1;
        clustFlags[i]     = -2; //setting cluster flag to -2, corresponds to coloring the node white in the CLR algorithm
    }

    if(rootmember == -1) {
        i_start = 0;
        i_end = n_total_particles;
    } else {
        i_start = rootmember;
        i_end = rootmember + 1;
    }
    // search for cluster roots
    for (i = i_start; i < i_end; i++) {
      // dereference
      p1 = &(partCfg[i]);

      if(p1->l.solid) {
          rootId = p1->p.identity;

          //if the particle is not already clustered, then it is a cluster
          //root.
          if( clustFlags[rootId] == -2 ) {

            //setting cluster id to -1
            //corresponds to coloring the node grey
            //in the algorithm from CLR
            clustFlags[rootId] = -1;

            //start a new cluster
            c = (particleCluster*) malloc(sizeof(particleCluster));
            init(c,rootId);

            //this queue stores the next particles to add to the cluster
            clusterQueueHead = rootId;
            clusterQueueTail = rootId;

            while( clusterQueueHead != -1 ) {

              p = clusterQueueHead;

              //label the immediate neighbours
              for( j = 0; j < p1->l.neb; j++ )
              {
                 // get neighbor ID
                 nebId = p1->l.neighbors[j];
                 // dereference
                 p2 = &(partCfg[nebId]);

                 if(p2->l.solid && clustFlags[nebId] == -2 && nebId != p) {

                     //mark this node 'grey'
                     clustFlags[nebId] = -1;

                     //add it to back of the queue
                     ParticleListP[clusterQueueTail] = nebId;
                     clusterQueueTail              = nebId;
                 }
              }

              //take the particle off the head of the queue
              //because all its neighbours should be on the queue now
              clusterQueueHead = ParticleListP[p];

              //add it fully to the cluster
              c->size++;

            }

            c->next = clust;
            clust   = c;
            clusterCount++;
            //fprintf(stderr,"clust: %d, clusterCount: %d\n", clust, clusterCount);
         }
       }
    }

    free(clustFlags);

    //fprintf(stderr,"clusterCount: %d \n", clusterCount);

    haveClusters = 1;

    return;

}


double* clusterSizeCount() {

        particleCluster *c;
        double      *csc;   // cluster size count
        int          total;

        csc   = (double*) malloc((solidParticles + 1)*sizeof(double));
        memset( csc, 0, (solidParticles + 1) * sizeof( double ) );
        total = 0;

        c = clust;
        while( c != 0 ){
          csc[c->size] += 1.0;
          total        += c->size;
          c = c->next;
        }

        if( total != solidParticles ){
            fprintf(stderr,"Some problem in cluster assignment\n");
            fprintf(stderr,"total particles: %d of %d \n", total, solidParticles);
        }

        return( csc );
}

double* writeClusterVec( int *pt_size ){

   double      *dVec;
   particleCluster *c;

  *pt_size = solidParticles + 1;

   if( !haveClusters ){
      buildClusters(-1);
   }

   dVec = clusterSizeCount();

   while( clust != 0 ){
       c     = clust->next;
       free(clust);
       clust = c;
   }
   haveClusters = 0;

   return( dVec );
}

int largestCluster() {

        particleCluster *c, *biggest;
        int          l;

        l = 0;
        if( !haveClusters ){
          buildClusters(-1);
        }
        // begin with last found cluster
        c = clust;
        // loop over all clusters (last one has 0 as next cluster)
        while( c != 0 ){
          if( c->size > l ){
              l = c->size;
              biggest = c;
          }
            c = c->next;
        }

        //buildClusters(biggest->root);

        return( l );
}
/** this reaction coordinate increases whenever two clusters of any size are joined */
int clusterMoment() {

        particleCluster *c;
        int          m;

        m = 0;

        if( !haveClusters ){
          buildClusters(-1);
        }
        c = clust;
        while( c != 0 ){
          m += ( c->size - 1 );
          c = c->next;
        }

        return( m );
}

// .::..::..::..::..::..::..::..::..::..::..::..::..::..::..::..::..::..::..::..::..::..::..::..::.

void visit_part(Particle *p, int clabel, int *tmp_size) {

    int c;

    /* Check if cell is already labelled */
    if(p->l.clabel == -1) {
        // label particle for current cluster if not labelled
        p->l.clabel = clabel;
        *tmp_size += 1;
        // Loop neighbors
        for(c = 0; c < p->l.neb; c++) {
            if(partCfg[p->l.neighbors[c]].l.solid) visit_part(&(partCfg[p->l.neighbors[c]]), clabel, tmp_size);
        }
    }
}

int kais_cluster() {

    int biggest_cluster = 0, biggest_cluster_label = -1;
    int i, tmp_size, clabel;
    Particle *p1;

    clabel = 0;
    tmp_size = 0;

    // init
    for (i = 0; i < n_total_particles; i++) {
        partCfg[i].l.clabel = -1;
    }

    // array of solid particles
    for (i = 0; i < n_total_particles; i++) {

        if(partCfg[i].l.solid) {
            // dereference
            p1 = &(partCfg[i]);

            visit_part(p1,clabel,&tmp_size);

            if (biggest_cluster < tmp_size) {
                biggest_cluster = tmp_size;
                biggest_cluster_label = clabel;
            }

            tmp_size = 0;
            clabel++;
        }
    }

    return biggest_cluster;
}

// .::..::..::..::..::..::..::..::..::..::..::..::..::..::..::..::..::..::..::..::..::..::..::..::.


/*
 Clustur algorithms done.
##################################################################################################
 Begin q6:
*/
#if 0
//spherical harmoic for l=6
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
	    p_tmp->l.q6r[m] += y6[0][m];
	    p_tmp->l.q6i[m] += y6[1][m];
    }

    //fprintf(stderr,"\n");
}
#endif
//berechnet q6!
int prepareQ6(double rc){

    //double rc = q6para.rc
    Particle *p1, *p2;
    int i, j, m;
    double dist2;
    double rclocal2 = rc*rc; // sphere radius squared around particle for neighbor detection
    double vec21[3];
    int statusOK = 1;
    
    for (i=0;i<n_total_particles;i++) {
            partCfg[i].l.neb=0;
            partCfg[i].l.solid = 0;
            partCfg[i].l.solidBonds = 0;
            partCfg[i].l.q6=0.0;
	        for (int m=0; m<=6; m++){
	            partCfg[i].l.q6r[m]=0.0;
	            partCfg[i].l.q6i[m]=0.0;
	        }
    } //n_total

    // outer loop
    for (i=0;i<n_total_particles;i++) {
	    p1 = &partCfg[i];                    // pointer to particle 1

        // inner loop
        for(j=0;j<n_total_particles;j++) {

            if (i != j) {
	            p2 = &partCfg[j];                  // pointer to particle 2

                dist2 = distance2vec(p2->r.p, p1->r.p, vec21);

                if(dist2 < rclocal2) {

                    if((p1->l.neb >= 127) || (p2->l.neb >= 127)) {
                        fprintf(stderr,"ERROR: Particle has more neighbors than possible!\n");
                    } else {
                        p1->l.neighbors[p1->l.neb]=p2->p.identity;

                        if(p2->p.identity != j) fprintf(stderr,"DANG!");

                        //p2->l.neighbors[p2->l.neb]=p1->p.identity;
                        p1->l.neb++;
                        //p2->l.neb++;
                    }
                    y6( p1, sqrt(dist2), vec21[0], vec21[1], vec21[2]);
                    //y6( p2, sqrt(dist2), vec21[0], vec21[1], vec21[2]);

                  }
            } // i != j
        } // j
    } // i


    totneb = 0;

    for (i=0;i<n_total_particles;i++) {

	        // Wolfgang Lechner and Christoph Dellago 2008 eq(1)
            if(partCfg[i].l.neb > 0) {
                for (m=0; m<=6; m++){
                    partCfg[i].l.q6r[m] /= (float) partCfg[i].l.neb;
                    partCfg[i].l.q6i[m] /= (float) partCfg[i].l.neb;
                }
            } else {
	            //Q6 undefined... system needs to collapse a little
	            //Q6 = 0.0;
	            partCfg[i].l.q6r[m] = 0.0;
	            partCfg[i].l.q6i[m] = 0.0;
	            statusOK = 0;
	        }

	        partCfg[i].l.q6 = 0.5 * ( partCfg[i].l.q6r[0] * partCfg[i].l.q6r[0] + partCfg[i].l.q6i[0] * partCfg[i].l.q6i[0] );
	        //fprintf(stderr, "Anfang: %f aus %f und %f\n", partCfg[i].l.q6, partCfg[i].l.q6r[0], partCfg[i].l.q6i[0]);
	        for (int m=1; m<=6; m++){
	            partCfg[i].l.q6 += partCfg[i].l.q6r[m] * partCfg[i].l.q6r[m] + partCfg[i].l.q6i[m] * partCfg[i].l.q6i[m];
	        }
	        //fprintf(stderr, "Ende: %f\n", partCfg[i].l.q6);
	        partCfg[i].l.q6 *= (4.0 * M_PI) / 13.0; //normalise by 4pi/13
         // Steinhardt order parameter: Wolfgang Lechner and Christoph Dellago 2008 eq(3)
	        partCfg[i].l.q6 = sqrt(partCfg[i].l.q6);    // This is the local invariant q6 per particle (Eq. 7 in ten Wolde)

            // Neigbor count
	        totneb += partCfg[i].l.neb;

            //fprintf(stderr,"Particle %d has %d neighbors. Q6: %f\n",partCfg[i].p.identity,partCfg[i].l.neb,partCfg[i].l.q6);
    }
    return statusOK;
}

int q6_calculation(){


    double rc = q6para.rc;
    Particle *p1, *p2;
    int i, j, m, n;
    double dist2;
    double rclocal2 = rc*rc; // sphere radius squared around particle for neighbor detection
    double vec21[3];
    int statusOK = 1;
    //totneb = 0;
//fprintf(stderr, "%d: rclocal2 %lf \n", this_node, rclocal2); 
    //int n_part;
    int np, np2;
    //int g, pnode;
    Cell *cell;
    int c;
    Particle *part, *part2;
    IA_Neighbor *neighbor;
    //MPI_Status status;
    
    //part on node
    //n_part = cells_get_n_particles();
    
    for (c = 0; c < local_cells.n; c++) {
      cell = local_cells.cell[c];
      part = cell->part;
      np = cell->n;
       
      for (i=0;i<np;i++) {
        part[i].l.neb=0;
        part[i].l.solid = 0;
        part[i].l.solidBonds = 0;
        part[i].l.q6=0.0;
	       for (int m=0; m<=6; m++){
	         part[i].l.q6r[m]=0.0;
	         part[i].l.q6i[m]=0.0;
	       }
      }
    }

      
    /* Loop local cells */
    for (c = 0; c < local_cells.n; c++) {
      cell = local_cells.cell[c];
      part = cell->part;
      np  = cell->n;
      
      for (i=0;i<np;i++) {
	       p1 = &part[i];	                        
        /* Loop cell neighbors */
        for (n = 0; n < dd.cell_inter[c].n_neighbors; n++) {
          neighbor = &dd.cell_inter[c].nList[n];
          part2  = neighbor->pList->part;
          np2 = neighbor->pList->n;
          for (j=0;j<np2;j++) {
            //if (i != j) {
	             p2 = &part2[j];
	             
              dist2 = distance2vec(p2->l.mean_pos, p1->l.mean_pos, vec21);
              //fprintf(stderr, "%d: dist2 %lf \n", this_node, dist2);
              if(dist2 < rclocal2) {
              #if 1
                if((p1->l.neb >= 127)) {
                   fprintf(stderr,"ERROR: Particle has more neighbors than possible! %i\n", p1->l.neb);
                     errexit();
                }
                #endif
                 //else {
                    p1->l.neighbors[p1->l.neb]=p2->p.identity;

                    //if(p2->p.identity != j){ fprintf(stderr,"DANG!");
                      //errexit();}
                    //p2->l.neighbors[p2->l.neb]=p1->p.identity;
                    p1->l.neb++;
                        //p2->l.neb++;
                  //}
                
                if(dist2 != 0.0) y6( p1, sqrt(dist2), vec21[0], vec21[1], vec21[2]);
                //if(dist2 != 0.0) y6( p2, sqrt(dist2), vec21[0], vec21[1], vec21[2]);
              }
            //} // i != j
          } // j
        } //cell neighbors
      } // i   
    }// local cells


    //totneb = 0;

    for (c = 0; c < local_cells.n; c++) {
      cell = local_cells.cell[c];
      part = cell->part;
      np = cell->n;
       
      for (i=0;i<np;i++) {
	       // Wolfgang Lechner and Christoph Dellago 2008 eq(1)
        if(part[i].l.neb > 0) {
          for (m=0; m<=6; m++){
            part[i].l.q6r[m] /= (float) part[i].l.neb;
            part[i].l.q6i[m] /= (float) part[i].l.neb;
            //fprintf(stderr,"Particle %d Q6r %f Q6: %f\n",part[i].p.identity,part[i].l.q6r[m],part[i].l.q6);
          }
        } else {
	           //Q6 undefined... system needs to collapse a little
	           //Q6 = 0.0;
	           part[i].l.q6r[m] = 0.0;
	           part[i].l.q6i[m] = 0.0;
	           statusOK = 0;
	         }

	       part[i].l.q6 = 0.5 * ( part[i].l.q6r[0] * part[i].l.q6r[0] + part[i].l.q6i[0] * part[i].l.q6i[0] );
	       //fprintf(stderr, "Anfang: %f aus %f und %f\n", part[i].l.q6, part[i].l.q6r[0], part[i].l.q6i[0]);
	       for (int m=1; m<=6; m++){
	         part[i].l.q6 += part[i].l.q6r[m] * part[i].l.q6r[m] + part[i].l.q6i[m] * part[i].l.q6i[m];
	       }
	       //fprintf(stderr, "Ende: %f\n", part[i].l.q6);
	       part[i].l.q6 *= (4.0 * M_PI) / 13.0; //normalise by 4pi/13
        // Steinhardt order parameter: Wolfgang Lechner and Christoph Dellago 2008 eq(3)
	       part[i].l.q6 = sqrt(part[i].l.q6);    // This is the local invariant q6 per particle (Eq. 7 in ten Wolde)
        //fprintf(stderr,"Particle %d has %d neighbors. Q6: %f\n",part[i].p.identity,part[i].l.neb,part[i].l.q6);
        // Neigbor count optional
	       //totneb += part[i].l.neb;
      }      
    } 

    return statusOK;
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
      
      for (i=0;i<np;i++) {
	       part[i].l.mean_pos[0] = (part[i].r.p[0]+part[i].l.mean_pos[0])/2;
	       part[i].l.mean_pos[1] = (part[i].r.p[1]+part[i].l.mean_pos[1])/2;
	       part[i].l.mean_pos[2] = (part[i].r.p[2]+part[i].l.mean_pos[2])/2;
	     }
	     
	   }
//printf("udate mean pos finished\n");
}

//parallization untested!!!
void reduceQ6(){

    double Q6r[7], Q6i[7]; //global Q6m, real and imaginary part
    double Q6;
    int i,m;

    // init
    for (m=0; m<=6; m++){
	    Q6r[m] = 0.0;
	    Q6i[m] = 0.0;
    }

    int np;
    Cell *cell;
    int c;
    Particle *part;

    for (c = 0; c < local_cells.n; c++) {
      cell = local_cells.cell[c];
      part = cell->part;
      np = cell->n;
       
      for (i=0;i<np;i++) {

    //for (i=0;i<n_total_particles;i++) {

	        //accumulate local q6 vector into global average
	        for (int m=0; m<=6; m++){
	            Q6r[m] += part[i].l.neb * part[i].l.q6r[m];
	            Q6i[m] += part[i].l.neb * part[i].l.q6i[m];
	        }
    //}
      }
    }  
    for(m = 0; m <= 6; m++ ) {
	    if( totneb > 0 ){
	        Q6r[m] /= totneb;
	        Q6i[m] /= totneb;
	    } else {
		    //Q6 is undefined
		    //fprintf(stderr,"undefined!");
            Q6r[m] = 0.0;
            Q6i[m] = 0.0;
	    }
    }

    // PR thesis eq 3.29 schilling
    Q6 = 0.5 * ( Q6r[0]*Q6r[0] + Q6i[0]*Q6i[0] );
    for (int m=1; m<=6; m++){
	    Q6 += Q6r[m]*Q6r[m] + Q6i[m]*Q6i[m];
    }
    Q6 *= 4.0 * M_PI;
    Q6 /= 13.0;
    Q6 = sqrt(Q6);

}


//helper function
inline double pair_q6q6( Particle *p, Particle *q ) {

    double q6q6;

    //fprintf(stderr,"Check %f %f\n",p->l.q6,q->l.q6);

    q6q6  = 0.5 * ( p->l.q6r[0] * q->l.q6r[0] + p->l.q6i[0] * q->l.q6i[0] );
    for (int m=1; m<=6; m++){
	    q6q6 += p->l.q6r[m] * q->l.q6r[m] + p->l.q6i[m] * q->l.q6i[m];
    }
    q6q6 /= ( p->l.q6 * q->l.q6 ); //why normalise by these factors? Tanja?
    q6q6 *= (4.0 * M_PI) / 13.0; //normalise by 4pi/13

    return( q6q6 );
}
//parallization does not work fo far due to ghostpart without q6!!!
double reduceQ6Q6(double q6q6_min, int min_solid_bonds){

  solidParticles = 0;

  //eQ6Q6 = 0.0;
  bondCount      = 0;
  
  int np, np2;
  Cell *cell;
  Particle *p1, *p2;
  int i, j, n, c;
  Particle *part, *part2;
  IA_Neighbor *neighbor;

    /* Loop local cells */
    for (c = 0; c < local_cells.n; c++) {
      cell = local_cells.cell[c];
      part = cell->part;
      np  = cell->n;
      
      for (i=0;i<np;i++) {
	       p1 = &part[i];
	       p1->l.q6q6 = 0.0;	                        
        /* Loop cell neighbors */
        for (n = 0; n < dd.cell_inter[c].n_neighbors; n++) {
          neighbor = &dd.cell_inter[c].nList[n];
          part2  = neighbor->pList->part;
          np2 = neighbor->pList->n;
          for (j=0;j<np2;j++) {
            //if (i != j) {
	             p2 = &part2[j];

              //fprintf(stderr,"Particle %d (q6: %f) has %d neighbors: %i \n",p1->p.identity,p1->l.q6,p1->l.neb,p1->l.neighbors[p1->l.neb]);
              
              if(p1->l.q6 !=0.0){
                 p1->l.q6q6 = pair_q6q6(p1, p2);
              } else p1->l.q6q6 = 0.0;
              //fprintf(stderr,"q6q6: %f\n", q6q6);

              //Test against arbitrary threshold
              if( p1->l.q6q6 > q6q6_min ) {
	               p1->l.solidBonds++;
              }

              //accumulate an average stat for the whole system
              //if( p1->l.neighbors[j] > i ) { //avoid double-counting
	             //  eQ6Q6 += p1->l.q6q6;
	             //  bondCount++;
              //}
            } 
        }// neighbor loop
            if( p1->l.solidBonds >= min_solid_bonds ){
	             p1->l.solid = 1;

	             solidParticles++;
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
#if 0
void preinit_q6(double tcl_rc, double tcl_q6q6_min, int tcl_min_solid_bonds){
  rc = tcl_rc;
  q6q6_min = tcl_q6q6_min;
  min_solid_bonds = tcl_min_solid_bonds;
}
#endif
/* init globals for q6 */
void init_q6() {
    clust = 0;
    haveClusters = 0;
    solidParticles = 0;
    totneb = 0;
    updatePartCfg(WITHOUT_BONDS);
    sortPartCfg();
}


double analyze_q6(double rc, double q6q6_min, int min_solid_bonds) {

    double avgQ6;
    // min_solid_bonds = 6
    // q6q6_min = 0.7

    init_q6();

    prepareQ6(rc);
    //reduceQ6();
    avgQ6 = reduceQ6Q6(q6q6_min, min_solid_bonds);


    freePartCfg();

    return avgQ6;
}

double analyze_q6_solid(double rc, double q6q6_min, int min_solid_bonds) {

    init_q6();

    prepareQ6(rc);
    //reduceQ6();
    reduceQ6Q6(q6q6_min, min_solid_bonds);

    freePartCfg();

    return solidParticles;
}

double analyze_q6_solid_cluster(double rc, double q6q6_min, int min_solid_bonds) {

    int biggest_cluster = 0;

    init_q6();

    prepareQ6(rc);
    //reduceQ6();
    solidParticles = reduceQ6Q6(q6q6_min, min_solid_bonds);

    biggest_cluster = kais_cluster();

    // alternative cluster algorithm
    //biggest_cluster_searchtree = largestCluster();

    freePartCfg();

    return biggest_cluster;
}

int initialize_q6(double tcl_rc, double tcl_q6q6_min, int tcl_min_solid_bonds) {

    q6para.rc = tcl_rc;
    q6para.q6q6_min = tcl_q6q6_min;
    q6para.min_solid_bonds = tcl_min_solid_bonds;
    mpi_bcast_q6_params();
    
    //printf("bcast q6 params ok\n");

    mpi_q6_calculation();
    
    //printf("mpi_q6_calculation ok\n");
    
    return 0;

}

void update_q6() {

    mpi_q6_calculation();

    //printf("mpi_q6_calculation update ok\n");

}

void q6_pre_init() {

    haveClusters = 0;
    clusterCount = 0;
    solidParticles = 0;
    totneb = 0;
    q6para.rc = 0.0;
    q6para.q6q6_min = 0.0;
    q6para.min_solid_bonds = 0;
    eQ6Q6 = 0.0;
    bondCount = 0;
    q6q6 = 0.0;
    
    reset_mean_part_pos();
    
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
	       part[i].l.mean_pos[0] = 0.0;
	       part[i].l.mean_pos[1] = 0.0;
	       part[i].l.mean_pos[2] = 0.0;
	     }
	     
	   }

}
#endif
