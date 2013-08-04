
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

/** \file cuda.hpp
 * Header file for all .cu files
 *
 * This is the header file for the Lattice Boltzmann implementation in lbgpu_cfile.c
 */

#ifndef CUDA_HPP
#define CUDA_HPP

#include <cuda.h>
/**erroroutput for memory allocation and memory copy
 * @param err cuda error code
 * @param *file .cu file were the error took place
 * @param line line of the file were the error took place
*/
extern cudaError_t err;
extern cudaError_t _err;

inline void _cuda_check_errors(cudaError_t err, char *file, unsigned int line)
{
#if 1
    if( cudaSuccess != err) {                                             
      fprintf(stderr, "Cuda Error at %s:%u.\n", file, line);
      printf("CUDA error: %s\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    } else {
      _err=cudaGetLastError(); \
      if (_err != cudaSuccess) {
        fprintf(stderr, "Error found during memory operation. Possibly however from an failed operation before. %s:%u.\n", file, line);
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
      }
    }
#endif
}
#define cuda_check_errors(a) _cuda_check_errors((a), __FILE__, __LINE__)
#define KERNELCALL(_f, _a, _b, _params) \
_f<<<_a, _b, 0, stream[g]>>>_params; \
_err=cudaGetLastError(); \
if (_err!=cudaSuccess){ \
  printf("CUDA error: %s\n", cudaGetErrorString(_err)); \
  fprintf(stderr, "error calling %s with #thpb %d in %s:%u\n", #_f, _b, __FILE__, __LINE__); \
  exit(EXIT_FAILURE); \
}

#endif /* LB_GPU_H */
