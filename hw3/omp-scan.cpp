#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <iostream>
using namespace std;


// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;

  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
  //for (long i=0; i<n; i++) cout << prefix_sum[i] << endl;
}

// ------------------------------------------------------------------------------------------//

void scan_omp(long* prefix_sum, const long* A, long n, long* T, int nthr) {
  // TODO: implement multi-threaded OpenMP scan
  T[0] =0;
  prefix_sum[0] = 0;
  if (n == 0 ) return;
  #pragma omp parallel num_threads(nthr)
  {
      long sum =0;
      int thr_num = omp_get_thread_num();
      #pragma omp for schedule(static)
      for (long i=0; i<n; i++){
          //cout << "A and threadnumber " << thr_num << "     " << A[i] << endl;
          sum += A[i];
          //cout << "Thread number " << thr_num << " has started with A= " << A[i] << " and sum as " << sum << endl;
          long tid = omp_get_thread_num();
          T[tid+1] = sum;
          prefix_sum[i+1] = sum;
      }
  }
  //cout<< " Printing thread-id " << endl;
  for (long i=0; i<nthr; i++){ 
  T[i+1] += T[i];
  //cout << T[i] << endl;
  }
  
  //cout<< " Now Printing Prefix sum " << endl;
  //for (long i=0; i<n; i++) cout << prefix_sum[i] << endl;
  
  #pragma omp parallel num_threads(nthr)
  {
    #pragma omp for schedule(static) 
      for (long i=1; i<n; i++){
        long tid = omp_get_thread_num();
        //cout << "Thread id is = " << tid << " operating on prefix[i] = "<< prefix_sum[i] << " and T[tid] is = " << T[tid] << endl;
        prefix_sum[i] += T[tid];
        //cout << " Final sum = " << prefix_sum[i] << T[tid] << endl;
      }
  }

  //for (long i=0; i<n; i++) cout << " Started printing prefix-sum " << prefix_sum[i] << endl;
}

int main() {
  long nthr[13] = {4, 8, 10, 16,32, 50, 64, 100, 128, 200, 256, 500, 1000};
  for (long i=0; i<13; i++){
    long N = 100000000;
    long* A = (long*) malloc(N * sizeof(long));
    long* B0 = (long*) malloc((N+1) * sizeof(long));
    long* B1 = (long*) malloc(N * sizeof(long));
    long* T1 = (long*) malloc(nthr[i] * sizeof(long));
    for (long i = 0; i < N; i++) A[i] = rand();
    //for (long i=0; i<N; i++) cout << A[i] << endl;
    cout << "    " << endl;
    cout << " The number of threads used are: " << nthr[i] << endl;
    double tt = omp_get_wtime();
    scan_seq(B0, A, N);
    printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

    tt = omp_get_wtime();
    scan_omp(B1, A, N, T1, nthr[i]);
    printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

    long err = 0;
    for (long i = 0; i < N; i++) err = max(err, abs(B0[i] - B1[i]));
    printf("error = %ld\n", err);

    free(A);
    free(B0);
    free(B1);
    free(T1);
  }
  return 0;
}