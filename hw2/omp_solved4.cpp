/******************************************************************************
* FILE: omp_bug4.c
* DESCRIPTION:
*   This very simple program causes a segmentation fault.
* AUTHOR: Blaise Barney  01/09/04
* LAST REVISED: 04/06/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1048

int main (int argc, char *argv[]) 
{
    int nthreads, i, tid, j;

    double** a = new double*[N];            /*** The major problem of the code was segmentation error. The code is running for N <300;
                                            As N increases to 1048, a huge chunk of memory is required which was not 
                                            allocated by the static 2D array. Thus by defining the array in a 2D-dynamic array form,
                                            the memory error could be solved for N with size 1048 ***/
    for (int k=0; k<N; k++){
        a[k] = new double[N];
    }
   
    /* Fork a team of threads with explicit variable scoping */
    #pragma omp parallel shared(nthreads) private(i,j,tid) 
    #pragma omp& firstprivate(a)
    {
        /* Obtain/print thread info */
        tid = omp_get_thread_num();
        if (tid == 0) 
        {
            nthreads = omp_get_num_threads();
            printf("Number of threads = %d\n", nthreads);
        }
        printf("Thread %d starting...\n", tid);
        /* Each thread works on its own private copy of the array */
        for (i=0; i<N; i++){
            for (j=0; j<N; j++){
                a[i][j] = tid + i + j;
            }
        }

        /* For confirmation */
        printf("Thread %d done. Last element= %f\n ", tid, a[N-1][N-1]);

    }  /* All threads join master thread and disband */
    	// Traverse the 2D array
   
    for (int i=0; i<N; i++) delete [] a[i];
    delete [] a;
}
