/******************************************************************************
* FILE: omp_bug2.c
* DESCRIPTION:
*   Another OpenMP program with a bug. 
* AUTHOR: Blaise Barney 
* LAST REVISED: 04/06/05 
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[]) 
{
    int nthreads, i, tid;
    double total;       /** The float type of "total" has been changed to double to avoid rounding-off the total **/
    total = 0.0;        /** The variable "total" was initialized to zero before the start of every thread calculation loop
                        essentially giving a final result of zero. Now this variable has been initialized before the parallel operation starts **/
    
    /*** Spawn parallel region ***/
    #pragma omp parallel 
    {
        /* Obtain thread number */
        tid = omp_get_thread_num();
        /* Only master thread does this */
        if (tid == 0) {
            nthreads = omp_get_num_threads();
            printf("Number of threads = %d\n", nthreads);
        }
        printf("Thread %d is starting...\n",tid);
    
        #pragma omp barrier
    
        /* do some work */
        #pragma omp for schedule(dynamic,10) reduction(+:total) /** The "reduction" command was missing **/
        for (i=0; i<1000000; i++){
            tid = omp_get_thread_num();
            total = total + i*1.0;
        }
        printf ("Thread %d is done! Total= %e\n",tid,total); /** The print statement **/
    } /*** End of parallel region ***/
    printf ("Thread %d Total= %e\n",tid,total); /** The print statement **/
}