#include <stdio.h>
#include <math.h>
#include "utils.h"
#include <algorithm>
#include <string>
using namespace std;
#define n_threads_per_block 1024          // To specify the number of threads per block
#define vec_size 1U << 18                 // pow(2, 18)

void Vec_Mul_CPU(long m, long n, const double *a, const double *b, double *c_ref) {
  for (long i = 0; i < m; i++){
    double sum = 0;
    for (long j = 0; j < n; j++){       // Matrix A stored in row-major format
      sum += a[j + n*i] * b[j];
    }
    c_ref[i] = sum;
  }
}

__global__ void vec_dot(long m, long n, const double *a, const double *b, double *c) {
  __shared__ double temp[n_threads_per_block];
  unsigned int tid = threadIdx.x;
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx<(n*m)) {
    //printf(" The threadid is %d, A is %f, b is %f \n", threadIdx.x, a[idx], b[idx%n]);
    temp[threadIdx.x] = a[idx] * b[idx%n];
  }
  __syncthreads();

  for (unsigned int s=1; s < blockDim.x; s *= 2) {
    int index = 2 * s * tid;
    if (index < blockDim.x) temp[index] += temp[index + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) c[blockIdx.x] = temp[0];
}


int main(int argc, char** argv){
  printf(" Dimension         Time in Cuda         Time in CPU                 GB/s           Error\n");

  //for (long p = 1024; p <= vec_size; p *= 2) 
  {
    long m = 1024, n = 1024, p = m*n;                            // m = rows, n = cols
    long n_blocks = (m*n)/n_threads_per_block;              // To specify the number of blocks
    // cout << "Number of blocks are " << n_blocks << endl; 

    double *a_dev, *b_dev, *c_dev;
    cudaMalloc((void**)&a_dev, m*n*sizeof(double));       // Matrix m*n
    cudaMalloc((void**)&b_dev, n*sizeof(double));         // Vector n*1
    cudaMalloc((void**)&c_dev, n_blocks*sizeof(double));         // Output m*1

    double* a = (double*) malloc(m*n*sizeof(double));         // Matrix A: m*n; stored in row-major format
    double* b = (double*) malloc(n*sizeof(double));           // Vector b: n*1
    double* c_ref = (double*) malloc(m*sizeof(double));       // Output m*1  
    double* c = (double*) malloc(n_blocks*sizeof(double));

    // Initialize the vectors
    for (long i = 0; i < m*n; i++) a[i] = rand() %100000;    //rand() %100000;
    for (long i = 0; i < n; i++) b[i] = rand() %100000;     //rand() %100000;
    for (long i = 0; i < m; i++) c_ref[i] = 0;
    
    Timer t;
    t.tic();
    Vec_Mul_CPU(m, n, a, b, c_ref);       // Compute reference solution on the CPU
    double time_vec_mul_cpu = t.toc();

    t.tic();
    cudaMemcpy(a_dev, a, m*n*sizeof(double), cudaMemcpyHostToDevice);         // Copying from Host to Device
    cudaMemcpy(b_dev, b, n*sizeof(double), cudaMemcpyHostToDevice);         // Copying from Host to Device
    
    vec_dot<<<n_blocks, n_threads_per_block>>>(m, n, a_dev, b_dev, c_dev);
    double time_vec_mul_gpu = t.toc();
    cudaMemcpy(c, c_dev, n_blocks*sizeof(double), cudaMemcpyDeviceToHost);       // Copying from Device to Host
  
    double bandwidth = ((m*n + n + n_blocks + n_blocks*n_threads_per_block) *4*sizeof(double))/(1000000000*time_vec_mul_gpu); 
    printf("%10d    %15f    %15f      %15f", p, time_vec_mul_gpu, time_vec_mul_cpu, bandwidth);

    long skip = ceil(n/n_threads_per_block);
    for (long i = 0; i < n; i++) {
      double final_sum = 0;
      for (long j = 0; j < skip; j++){
        final_sum += c[j + skip*i];
      }
      c[i] = final_sum;
    } 
    
    double max_err = 0;
    for (long i = 0; i < m; i++) 
    {
      if(max_err < fabs(c[i] - c_ref[i]))
      max_err = fabs(c[i] - c_ref[i]);
    }
    printf(" %15e\n", max_err);
   
    cudaFree(a_dev);
    cudaFree(b_dev);
    cudaFree(c_dev);
    
    free(a);
    free(b);
    free(c);
    free(c_ref);
  }
  return 0;
}
