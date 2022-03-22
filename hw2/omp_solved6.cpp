/******************************************************************************
* FILE: omp_bug6.c
* DESCRIPTION:
*   This program compiles and runs fine, but produces the wrong result.
*   Compare to omp_orphan.c.
* AUTHOR: Blaise Barney  6/05
* LAST REVISED: 06/30/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define VECLEN 5

float a[VECLEN], b[VECLEN];
float sum;                          /*** The major error in this code was the scope of the variable sum which was not global.
                                        This scope of the variable is made global by defining it here. ***/

float dotprod (float sum)
{
    int i, tid;
    tid = omp_get_thread_num();
    #pragma omp& for reduction(+:sum)
        for (i=0; i < VECLEN; i++)
        {
            sum = sum + (a[i]*b[i]);
            printf("  tid= %d i=%d\n",tid,i);
        }
    return 0;                           /*** A return statement added for the float type function ***/
}


int main (int argc, char *argv[]) 
{
    int i;
    float sum;

    for (i=0; i < VECLEN; i++)
        a[i] = b[i] = 1.0 * i;
    sum = 0.0;

    #pragma omp parallel shared(sum)
    dotprod(sum);                       /*** The shared variable "sum" is passed to the function "dotprod" ***/

    return 0;

}
