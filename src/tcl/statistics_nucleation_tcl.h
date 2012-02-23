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

/** \file statistics_nucleation.h
 *
 *  This file contains algorithms useful for nucleation.
 *  It can be used to identify the largest cluster of neighbor particles and
 *  for detecting whether particles are solid-like or not. Therefore, the Q6 algorithm is provided.
 *  
 *
 */

#ifndef STATISTICS_NUCLEATION_TCL_H
#define STATISTICS_NUCLEATION_TCL_H

#include "parser.h"

/** Largest bubble volume of vapor-like particles
*/
int tclcommand_analyze_bubble_volume(Tcl_Interp *interp, int argc, char **argv);

/** Local bond order parameter, returns system average
*/
int tclcommand_analyze_q6(Tcl_Interp *interp, int argc, char **argv);

/** Identify by q6, how many solid particles there are in the system
*/
int tclcommand_analyze_q6_solid(Tcl_Interp *interp, int argc, char **argv);

/** Return size of largest solid cluster, identified by q6
*/
int tclcommand_analyze_q6_solid_cluster(Tcl_Interp *interp, int argc, char **argv);

#endif
