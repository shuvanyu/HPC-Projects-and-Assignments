#include <stdio.h>
#include <math.h>
#include "utils.h"
#include <algorithm>
#include <string>
using namespace std;
#define n_threads_per_block 1024          // To specify the number of threads per block
#define vec_size 1U << 18                 // pow(2, 18)

void Vec_Mul_CPU(long p, const double *a, const double *b, double &c_ref) {
  for (long i = 0; i < p; i++) c_ref += a[i] * b[i];
}

__global__ void vec_dot(long p, const double *a, const double *b, double *c) {
  __shared__ double temp[n_threads_per_block];
  unsigned int tid = threadIdx.x;
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx<p) temp[threadIdx.x] = a[idx] * b[idx];
  __syncthreads();

  for (unsigned int s=1; s < blockDim.x; s *= 2) {
  int index = 2 * s * tid;
  if (index < blockDim.x) temp[index] += temp[index + s];
  __syncthreads();
  }
  if (threadIdx.x == 0) c[blockIdx.x] = temp[0];
}


int main(int argc, char** argv){
  printf(" Dimension         Time in Cuda         Time in CPU          Gflop/s          GB/s           Error\n");

  for (long p = 1024; p <= vec_size; p *= 2) 
  {
    long n_blocks = p/n_threads_per_block;              // To specify the number of blocks 

    double *a_dev, *b_dev, *c_dev;
    cudaMalloc((void**)&a_dev, p*sizeof(double));
    cudaMalloc((void**)&b_dev, p*sizeof(double));
    cudaMalloc((void**)&c_dev, n_blocks*sizeof(double));

    double* a = (double*) malloc(p*sizeof(double)); 
    double* b = (double*) malloc(p*sizeof(double));
    double* c = (double*) malloc(n_blocks*sizeof(double));

    // Initialize the vectors
    for (long i = 0; i < p; i++) a[i] = rand() %100000;
    for (long i = 0; i < p; i++) b[i] = rand() %100000;
    
    Timer t;
    t.tic();
    double c_ref = 0.;
    Vec_Mul_CPU(p, a, b, c_ref);       // Compute reference solution on the CPU
    double time_vec_mul_cpu = t.toc();

    t.tic();
    cudaMemcpy(a_dev, a, p*sizeof(double), cudaMemcpyHostToDevice);         // Copying from Host to Device
    cudaMemcpy(b_dev, b, p*sizeof(double), cudaMemcpyHostToDevice);         // Copying from Host to Device

    vec_dot<<<n_blocks, n_threads_per_block>>>(p, a_dev, b_dev, c_dev);
    double time_vec_mul_gpu = t.toc();
    cudaMemcpy(c, c_dev, n_blocks*sizeof(double), cudaMemcpyDeviceToHost);       // Copying from Device to Host
  
    double flops = (2*p*p)/(1000000000*time_vec_mul_gpu); 
    double bandwidth = (2*p*p*sizeof(rand()))/(1000000000*time_vec_mul_gpu); 
    printf("%10d    %15f    %15f    %15f     %15f", p, time_vec_mul_gpu, time_vec_mul_cpu, flops, bandwidth);

    double final_sum = 0;
    for (long i = 0; i < n_blocks; i++) final_sum += c[i];

    double max_err = fabs(final_sum - c_ref);
    printf(" %15e\n", max_err);
   
    cudaFree(a_dev);
    cudaFree(b_dev);
    cudaFree(c_dev);
    
    free(a);
    free(b);
    free(c);
  }
  return 0;
}
