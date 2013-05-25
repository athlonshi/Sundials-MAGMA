/*
 * -----------------------------------------------------------------
 * $Revision: 1.13 $
 * $Date: 2010/12/01 22:10:38 $
 * ----------------------------------------------------------------- 
 * Programmer(s): Scott D. Cohen, Alan C. Hindmarsh, Radu Serban
 *                and Dan Shumaker @ LLNL
 * -----------------------------------------------------------------
 * Copyright (c) 2002, The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * All rights reserved.
 * For details, see the LICENSE file.
 * -----------------------------------------------------------------
 * This is the interface file for the main CVODE integrator.
 * -----------------------------------------------------------------
 *
 * CVODE is used to solve numerically the ordinary initial value
 * problem:
 *
 *                 y' = f(t,y),
 *                 y(t0) = y0,
 *
 * where t0, y0 in R^N, and f: R x R^N -> R^N are given.
 *
 * -----------------------------------------------------------------
 */

#include <stdlib.h>
#include <stdarg.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include "magma.h"

#include <sundials/sundials_nvector.h>
#include <sundials/sundials_direct.h>

/*
 * -----------------------------------------------------------------
 * Function : CVodeInitGPU
 * -----------------------------------------------------------------
 * CVodeInit allocates and initializes memory (GPU) for a problem to
 * to be solved by CVODE.
 *
 * cvode_mem is pointer to CVODE memory returned by CVodeCreate.
 *
 * type is 0 to use CPU LU solver 1 to use GPU LU solver
 *
 * Return flag:
 *  CV_SUCCESS if successful
 *  CV_MEM_NULL if the cvode memory was NULL
 *  CV_MEM_FAIL if a memory allocation failed
 *  CV_ILL_INPUT f an argument has an illegal value.
 * -----------------------------------------------------------------
 */

#ifdef __cplusplus  /* wrapper to enable C++ usage */
extern "C" {
#endif

SUNDIALS_EXPORT int CVodeInitGPU(void *cvode_mem, int type);
SUNDIALS_EXPORT long int DenseGETRFGPU(DlsMat A, long int *p);

#define MAGMA_CUDA_INIT()                                                  \
    if( CUBLAS_STATUS_SUCCESS != cublasInit() ) {                          \
        fprintf(stderr, "ERROR: cublasInit failed\n");                     \
        exit(-1);                                                          \
    }                                                                      \
    printout_devices();                                                    \

#define MAGMA_CUDA_FINALIZE()                                              \
    cublasShutdown();

#define MAGMA_MALLOC( ptr, type, size )                                    \
    if ( MAGMA_SUCCESS !=                                                  \
            magma_malloc_cpu( (void**) &ptr, (size)*sizeof(type) )) {      \
        fprintf( stderr, "!!!! malloc failed for: %s\n", #ptr );           \
        exit(-1);                                                          \
    }

#define MAGMA_HOSTALLOC( ptr, type, size )                                    \
    if ( MAGMA_SUCCESS !=                                                     \
            magma_malloc_pinned( (void**) &ptr, (size)*sizeof(type) )) {      \
        fprintf( stderr, "!!!! magma_malloc_pinned failed for: %s\n", #ptr ); \
        exit(-1);                                                             \
    }

#define MAGMA_DEVALLOC( ptr, type, size )                                  \
    if ( MAGMA_SUCCESS !=                                                  \
            magma_malloc( (void**) &ptr, (size)*sizeof(type) )) {          \
        fprintf( stderr, "!!!! magma_malloc failed for: %s\n", #ptr );     \
        exit(-1);                                                          \
    }

#define MAGMA_FREE(ptr)                                                  \
    magma_free_cpu(ptr);

#define MAGMA_HOSTFREE(ptr)                                              \
    magma_free_pinned( ptr );

#define MAGMA_DEVFREE(ptr)                                               \
    magma_free( ptr );

/*Define global variable allocated to GPU*/
double *d_A ;

#ifdef __cplusplus  /* wrapper to enable C++ usage */
}
#endif

