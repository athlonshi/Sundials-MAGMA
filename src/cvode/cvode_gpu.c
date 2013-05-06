/*
 * -----------------------------------------------------------------
 * $Revision: 1.24 $
 * $Date: 2012/03/06 21:58:36 $
 * -----------------------------------------------------------------
 * Programmer(s): Scott D. Cohen, Alan C. Hindmarsh, Radu Serban,
 *                and Dan Shumaker @ LLNL
 * -----------------------------------------------------------------
 * Copyright (c) 2002, The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * All rights reserved.
 * For details, see the LICENSE file.
 * -----------------------------------------------------------------
 * This is the implementation file for the main CVODE integrator.
 * It is independent of the CVODE linear solver in use.
 * -----------------------------------------------------------------
 */

/*=================================================================*/
/*             Import Header Files                                 */
/*=================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

#include "cvode_impl.h"
#include <cvode/cvode_gpu.h>
#include <sundials/sundials_dense.h>
#include <sundials/sundials_math.h>
#include <sundials/sundials_types.h>

int CVodeInitGPU(void *cvode_mem, int type)
{
  CVodeMem cv_mem;

  /* Check cvode_mem */

  if (cvode_mem==NULL) {
    CVProcessError(NULL, CV_MEM_NULL, "CVODE", "CVodeInitGPU", MSGCV_NO_MEM);
    return(CV_MEM_NULL);
  }
  cv_mem = (CVodeMem) cvode_mem;

  /* Check for legal input parameters */
  if (type != 0 && type != 1) {
    CVProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "CVodeInitGPU", MSGCV_TYPE);
    return(CV_ILL_INPUT);
  }
  
  if (type == 1) {
    cv_mem->GPU = TRUE;
    MAGMA_CUDA_INIT();
  }
  else {
    cv_mem->GPU = FALSE;
  } 

  return(CV_SUCCESS);
}

long int DenseGETRFGPU(DlsMat A, long int *p)
{ 
  int M = A->M;
  int N = A->N;
  int lda = M;
  int ldda = ((M+31)/32)*32;
  int info;
  int i, j;
  int p1[M];
  
/*Copy to a temporary CPU*/ 
  for(i=0; i<M; i++)
    for(j=0; j<N; j++) h_A[j*M+i] = DENSE_ELEM(A,i,j);

/*Call MAGMA LU factorization solver*/
  magma_dsetmatrix( M, N, h_A, lda, d_A, ldda ); 
  magma_dgetrf_gpu( M, N, d_A, ldda, p1, &info);
  magma_dgetmatrix( M, N, d_A, ldda, h_A, lda );

/*Put back to CPU*/
  for(i=0; i<M; i++) {
    p[i] = p1[i]-1 ;
    for(j=0; j<N; j++) {
      DENSE_ELEM(A,i,j) = h_A[j*M+i];
    }
  }

  return(info);

}