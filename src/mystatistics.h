#ifndef MY_STATISTICS_H
#define MY_STATISTICS_H
#include "utils.h"
#include <tcl.h>

extern int parse_wallstuff(Tcl_Interp *interp, int argc, char **argv);

#ifdef MY_STAT

void save_sets();
extern int num_of_sets;
#endif

#endif
