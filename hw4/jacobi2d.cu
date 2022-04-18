#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
using namespace std;


void jacobi2d_cpu(double* u_new_jac, double* u_jac, double* f, const int n_col, const int n_grid, const double h){
    for(long row = 1; row < n_grid+1; ++row){
        for(long col = 1; col < n_grid+1; ++col){
            u_new_jac[row*n_col+col] = 0.25*(h*h*f[row*n_col+col] + u_jac[(row-1)*n_col+col] + u_jac[row*n_col+(col-1)] +
                                                        u_jac[(row+1)*n_col+col] + u_jac[row*n_col+(col+1)]);
        }
    }
}


__global__
void jacobi2d_gpu(double* u_cuda, double* u_new_cuda, const double* f, double* u_error, const long N, const long n_col, const double h){
    int col_idx = 1 + blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx =1 +  blockIdx.y * blockDim.y + threadIdx.y;
    int idx = row_idx * n_col + col_idx;

    if(idx < N){
    u_new_cuda[idx] = 0.25*(h*h*f[row_idx*n_col+col_idx] + u_cuda[(row_idx-1)*n_col+col_idx] + u_cuda[(row_idx+1)*n_col+col_idx] +
                        u_cuda[(row_idx)*n_col+col_idx+1] + u_cuda[(row_idx)*n_col+col_idx-1]);
    __syncthreads();
    }
    u_error[idx] = fabs(u_new_cuda[idx] - u_cuda[idx]);
    u_cuda[idx] = u_new_cuda[idx];
}


int main(int argc, char** argv) {
    int n_grid = 1024;
    int n_row = n_grid + 2;
    int n_col = n_grid + 2;
    int N = n_row * n_col;
    double h = 1./(n_grid + 1);
    printf("Number of discretization points: %d\n",n_grid);

    double* u = (double*) malloc(N * sizeof(double));
    double* u_new = (double*) malloc(N * sizeof(double));
    double* f = (double*) malloc(N * sizeof(double));
    double* u_new_cuda = (double*) malloc(N * sizeof(double));
    double* u_error = (double*) malloc(N * sizeof(double));

    for(int i = 0; i < N; ++i){
        u[i] = 0.;
        u_new[i] = 0.;
        f[i] = 1.;
        u_new_cuda[i] = 0.;
    }

    double *u_dev, *u_new_dev, *f_dev, *u_error_cuda;

    cudaMalloc(&u_dev, N*sizeof(double));   
    cudaMalloc(&u_new_dev, N*sizeof(double));
    cudaMalloc(&f_dev, N*sizeof(double));
    cudaMalloc(&u_error_cuda, N*sizeof(double));

    cudaMemcpy(u_dev, u, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(f_dev, f, N*sizeof(double), cudaMemcpyHostToDevice);

    int n_blocks_x = 32;
    int n_blocks_y = 32;
    dim3 blockShape(n_blocks_x,n_blocks_y);
    dim3 gridShape(n_grid/n_blocks_x, n_grid/n_blocks_y);

    cout << "Gird Shape: " << "Blocks in x_dir: " <<  n_grid/n_blocks_x << " Blocks in y_dir: " << n_grid/n_blocks_y << 
                 " | Block Shape: " << "Threads in x_dir: " << n_blocks_x << " Threads in y_dir: " << n_blocks_y << endl;
    
    double last_iter_err = 0.;
    cout << "Iteration Number      " << "Error (CPU - Cuda)        " << "Jacobi Residual on each iteration" << endl;
    for(int iter = 0; iter < 50; ++iter){
        double maxError = -1;
        jacobi2d_gpu<<<gridShape,blockShape>>>(u_dev, u_new_dev, f_dev, u_error_cuda, N, n_col, h);
        cudaDeviceSynchronize();
        cudaMemcpy(u_error, u_error_cuda, N*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(u_new_cuda, u_new_dev, N*sizeof(double), cudaMemcpyDeviceToHost);

        jacobi2d_cpu(u_new, u, f, n_col, n_grid, h);

        double err_cuda_cpu = -1;
        for(int i = 0; i < N; ++i){
            double diff = u_new_cuda[i] - u_new[i];
            if(diff > err_cuda_cpu){
                err_cuda_cpu = diff;
            }
        }
        iter++;

        for(long i = 0; i < N; ++i) u[i] = u_new[i];

        for(int i = 0; i < N; ++i){
            if(maxError < u_error[i]){
                maxError = u_error[i];
            }
        }
        last_iter_err = maxError;
        cout << "       "<< iter << "                    "<< err_cuda_cpu << "                "<< last_iter_err << endl;
    }

    cudaFree(u_dev);
    cudaFree(u_new_dev);
    cudaFree(f_dev);
    cudaFree(u_error_cuda);

    free(u);
    free(u_new);
    free(f);
    free(u_new_cuda);
    free(u_error);
    return 0;
}