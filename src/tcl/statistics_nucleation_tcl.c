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

#include <tcl.h>
#include "communication.h"
#include "parser.h"
#include "statistics_nucleation.h"

/* Main, by-TCL-called function */
int tclcommand_analyze_bubble_volume(Tcl_Interp *interp, int argc, char **argv)
{
    char buffer[TCL_DOUBLE_SPACE];
    double bubble_cut; // e.g. 1.0
    double sigma; // e.g. 1.0

    if (n_nodes > 1) {
        Tcl_AppendResult(interp, "Error: Largest bubble volume can only be calculated on a single processor", (char *)NULL);
        return TCL_ERROR;
    }

    /* check parameter types */
    if( (! ARG_IS_D(0, bubble_cut)) ||
        (! ARG_IS_D(1, sigma)) ) {
        Tcl_AppendResult(interp, "Error: Usage: analyze bubble_volume bubble_cut.\n", (char *)NULL);
        return TCL_ERROR;
    }

    Tcl_PrintDouble(interp, analyze_bubble_volume(interp,bubble_cut,sigma), buffer);
    Tcl_AppendResult(interp, buffer, (char *)NULL);

    return TCL_OK;
}


int tclcommand_analyze_q6(Tcl_Interp *interp, int argc, char **argv) {

    char buffer[TCL_DOUBLE_SPACE];
    double tcl_rc, tcl_q6q6_min;
    int tcl_min_solid_bonds;

    if (n_nodes > 1) {
        Tcl_AppendResult(interp, "Error: Q6 can only be calculated on a single processor", (char *)NULL);
        return TCL_ERROR;
    }

    /* check parameter types */
    if( (! ARG_IS_D(0, tcl_rc)) ||
        (! ARG_IS_D(1, tcl_q6q6_min))  ||
        (! ARG_IS_I(2, tcl_min_solid_bonds)) ) {
        Tcl_AppendResult(interp, "Error: Usage: analyze q6 rc_neighbor q6q6_min min_solid_bonds.\n", (char *)NULL);
        return TCL_ERROR;
    }

    Tcl_PrintDouble(interp, analyze_q6(tcl_rc, tcl_q6q6_min, tcl_min_solid_bonds), buffer);
    Tcl_AppendResult(interp, buffer, (char *)NULL);

    return TCL_OK;
}

int tclcommand_analyze_q6_solid(Tcl_Interp *interp, int argc, char **argv) {

    char buffer[TCL_DOUBLE_SPACE];
    double tcl_rc, tcl_q6q6_min;
    int tcl_min_solid_bonds;

#if 0
    if (n_nodes > 1) {
        Tcl_AppendResult(interp, "Error: Q6 can only be calculated on a single processor", (char *)NULL);
        return TCL_ERROR;
    }
#endif
    /* check parameter types */
    if( (! ARG_IS_D(0, tcl_rc)) ||
        (! ARG_IS_D(1, tcl_q6q6_min))  ||
        (! ARG_IS_I(2, tcl_min_solid_bonds)) ) {
        Tcl_AppendResult(interp, "Error: Usage: analyze q6 rc_neighbor q6q6_min min_solid_bonds.\n", (char *)NULL);
        return TCL_ERROR;
    }

    Tcl_PrintDouble(interp, analyze_q6_solid(tcl_rc, tcl_q6q6_min, tcl_min_solid_bonds), buffer);
    Tcl_AppendResult(interp, buffer, (char *)NULL);

    return TCL_OK;
}

int tclcommand_analyze_q6_solid_cluster(Tcl_Interp *interp, int argc, char **argv) {

    char buffer[TCL_DOUBLE_SPACE];
    double tcl_rc, tcl_q6q6_min;
    int tcl_min_solid_bonds;
#if 0
    if (n_nodes > 1) {
        Tcl_AppendResult(interp, "Error: Q6 can only be calculated on a single processor", (char *)NULL);
        return TCL_ERROR;
    }
#endif
    /* check parameter types */
    if( (! ARG_IS_D(0, tcl_rc)) ||
        (! ARG_IS_D(1, tcl_q6q6_min))  ||
        (! ARG_IS_I(2, tcl_min_solid_bonds)) ) {
        Tcl_AppendResult(interp, "Error: Usage: analyze q6 rc_neighbor q6q6_min min_solid_bonds.\n", (char *)NULL);
        return TCL_ERROR;
    }

    Tcl_PrintDouble(interp, analyze_q6_solid_cluster(tcl_rc, tcl_q6q6_min, tcl_min_solid_bonds), buffer);
    Tcl_AppendResult(interp, buffer, (char *)NULL);

    return TCL_OK;
}

int tclcommand_q6(ClientData data, Tcl_Interp *interp, int argc, char **argv) {
#ifdef Q6_PARA
  argc--; argv++;
  
  double doublearg[2];
  int intarg;

  if (argc < 1) {
    Tcl_AppendResult(interp, "Error: q6 needs at least 1 argument\n", (char *)NULL);
    Tcl_AppendResult(interp, "Usage of \"q6\":\n", (char *)NULL);
    Tcl_AppendResult(interp, "Usage: q6 initialize rc_neighbor q6q6_min min_solid_bonds or q6 update\n", (char *)NULL);
    return TCL_ERROR;
  }
  while (argc > 0) {
    if (ARG0_IS_S("initialize")) { 
       argc--; argv++;
      /* check parameter types */
      if( (! ARG_IS_D(0, doublearg[0])) ||
          (! ARG_IS_D(1, doublearg[1]))  ||
          (! ARG_IS_I(2, intarg)) ) {
          Tcl_AppendResult(interp, "Error: Usage: q6 initialize rc_neighbor q6q6_min min_solid_bonds.\n", (char *)NULL);
          return TCL_ERROR;
      }
      if ( !initialize_q6(doublearg[0], doublearg[1], intarg) == 0 ) {
	          Tcl_AppendResult(interp, "Unknown Error set q6 paramters", (char *)NULL);
            return TCL_ERROR;
      }
    }
    else if (ARG0_IS_S("update")) {
        update_q6();
    }
    return TCL_OK;
  }
  return TCL_OK;
#else
  Tcl_AppendResult(interp, "Q6 is not compiled in!", NULL);
  return TCL_ERROR;
#endif
}


