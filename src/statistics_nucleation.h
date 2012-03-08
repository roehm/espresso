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

int q6_calculation(double dummy);

double analyze_bubble_volume(Tcl_Interp *interp, double bubble_cut, double sigma);

double analyze_q6(double tcl_rc, double tcl_q6q6_min, int tcl_min_solid_bonds);

double analyze_q6_solid(double tcl_rc, double tcl_q6q6_min, int tcl_min_solid_bonds);

double analyze_q6_solid_cluster(double tcl_rc, double tcl_q6q6_min, int tcl_min_solid_bonds);

int initialize_q6(double tcl_rc, double tcl_q6q6_min, int tcl_min_solid_bonds);

void update_q6();
#endif
