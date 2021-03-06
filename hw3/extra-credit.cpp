//g++ -std=c++11 -O3 -march=native -fno-tree-vectorize t.cpp -o t

#include <math.h>
#include <stdio.h>
#include "utils.h"
#include "intrin-wrapper.h"

// Headers for intrinsics
#ifdef __SSE__
#include <xmmintrin.h>
#endif
#ifdef __SSE2__
#include <emmintrin.h>
#endif
#ifdef __AVX__
#include <immintrin.h>
#endif

// coefficients in the Taylor series expansion of sin(x)
static constexpr double c3  = -1/(((double)2)*3);
static constexpr double c5  =  1/(((double)2)*3*4*5);
static constexpr double c7  = -1/(((double)2)*3*4*5*6*7);
static constexpr double c9  =  1/(((double)2)*3*4*5*6*7*8*9);
static constexpr double c11 = -1/(((double)2)*3*4*5*6*7*8*9*10*11);
// sin(x) = x + c3*x^3 + c5*x^5 + c7*x^7 + x9*x^9 + c11*x^11

// coefficients in the Taylor series expansion of cos(x)
static constexpr double c2  = -1/(((double)2));
static constexpr double c4  =  1/(((double)2)*3*4);
static constexpr double c6  = -1/(((double)2)*3*4*5*6);
static constexpr double c8  =  1/(((double)2)*3*4*5*6*7*8);
static constexpr double c10 = -1/(((double)2)*3*4*5*6*7*8*9*10);
static constexpr double c12 =  1/(((double)2)*3*4*5*6*7*8*9*10*11*12);
// cos(x) = 1 + c2*x^2 + c4*x^4 + c6*x^6 + c8*x^6 + c10*x^10 + c12*x^12

void sin4_reference(double* sinx, const double* x) {
  for (long i = 0; i < 4; i++) sinx[i] = sin(x[i]);
}

void cos4_reference(double* cosx, const double* x) {
  for (long i = 0; i < 4; i++) cosx[i] = cos(x[i]);
}

void sin4_taylor(double* sinx, const double* x_mod) {
  for (int i = 0; i < 4; i++) {
    double x1  = x_mod[i];
    double x2  = x1 * x1;
    double x3  = x1 * x2;
    double x5  = x3 * x2;
    double x7  = x5 * x2;
    double x9  = x7 * x2;
    double x11 = x9 * x2;

    double s = x1;
    s += x3  * c3;
    s += x5  * c5;
    s += x7  * c7;
    s += x9  * c9;
    s += x11 * c11;
    
    sinx[i] = s;
  }
}
// ------------------------------------ End of Function ---------------------------------- //    

void cos4_taylor(double* cosx, const double* x_mod) {
  for (int i = 0; i < 4; i++) {
    double x1  = x_mod[i];
    double x2  = x1 * x1;
    double x4  = x2 * x2;
    double x6  = x4 * x2;
    double x8  = x6 * x2;
    double x10 = x8 * x2;
    double x12 = x10 * x2;

    double s = 1;
    s += x2  * c2;
    s += x4  * c4;
    s += x6  * c6;
    s += x8  * c8;
    s += x10 * c10;
    s += x12 * c12;
    
    cosx[i] = s;
  }
}
// ------------------------------------ End of Function ---------------------------------- //    

void sin4_intrin(double* sinx, const double* x_mod) {
  // The definition of intrinsic functions can be found at:
  // https://software.intel.com/sites/landingpage/IntrinsicsGuide/#
#if defined(__AVX__)
  __m256d x1, x2, x3;
  x1  = _mm256_load_pd(x);
  x2  = _mm256_mul_pd(x1, x1);
  x3  = _mm256_mul_pd(x1, x2);

//  std::cout << "Inside AVX Function" << std::endl;

  __m256d s = x1;
  s = _mm256_add_pd(s, _mm256_mul_pd(x3 , _mm256_set1_pd(c3 )));
  _mm256_store_pd(sinx, s);
#elif defined(__SSE2__)

//  std::cout << "Inside SSE Function" << std::endl;

   constexpr int sse_length = 2;
  for (int i = 0; i < 4; i+=sse_length) {
    __m128d x1, x2, x3, x5,x7, x9, x11;
    x1  = _mm_load_pd(x_mod+i);
    x2  = _mm_mul_pd(x1, x1);
    x3  = _mm_mul_pd(x1, x2);
    x5  = _mm_mul_pd(x3, x2);
    x7  = _mm_mul_pd(x5, x2);
    x9  = _mm_mul_pd(x7, x2);
    x11  = _mm_mul_pd(x9, x2);
    __m128d s = x1;
    s = _mm_add_pd(s, _mm_mul_pd(x3 , _mm_set1_pd(c3 )));
    s = _mm_add_pd(s, _mm_mul_pd(x5 , _mm_set1_pd(c5 )));
    s = _mm_add_pd(s, _mm_mul_pd(x7 , _mm_set1_pd(c7 )));
    s = _mm_add_pd(s, _mm_mul_pd(x9 , _mm_set1_pd(c9 )));
    s = _mm_add_pd(s, _mm_mul_pd(x11 , _mm_set1_pd(c11 )));
    _mm_store_pd(sinx+i, s);
  }
#else
  sin4_reference(sinx, x);
#endif
}


// ------------------------------------ End of Function ---------------------------------- //

void cos4_intrin(double* cosx, const double* x_mod) {
  // The definition of intrinsic functions can be found at:
  // https://software.intel.com/sites/landingpage/IntrinsicsGuide/#
#if defined(__AVX__)
  __m256d x0, x1, x2;
  x0  = _mm256_set1_pd(1.);
  x1  = _mm256_load_pd(x);
  x2  = _mm256_mul_pd(x1, x1);

//  std::cout << "Inside AVX Function" << std::endl;

  __m256d s = x0;
  s = _mm256_add_pd(s, _mm256_mul_pd(x2 , _mm256_set1_pd(c2 )));
  _mm256_store_pd(cosx, s);
#elif defined(__SSE2__)

//  std::cout << "Inside SSE Function" << std::endl;

   int sse_length = 2;
  for (int i = 0; i < 4; i+=sse_length) {
    __m128d x0, x1, x2, x4, x6,x8, x10;
    x0  = _mm_set1_pd (1.);
    x1  = _mm_load_pd(x_mod+i);
    x2  = _mm_mul_pd(x1, x1);
    x4  = _mm_mul_pd(x2, x2);
    x6  = _mm_mul_pd(x4, x2);
    x8  = _mm_mul_pd(x6, x2);
    x10  = _mm_mul_pd(x8, x2);
    __m128d s = x0;

    s = _mm_add_pd(s, _mm_mul_pd(x2 , _mm_set1_pd(c2 )));
    s = _mm_add_pd(s, _mm_mul_pd(x4 , _mm_set1_pd(c4 )));
    s = _mm_add_pd(s, _mm_mul_pd(x6 , _mm_set1_pd(c6 )));
    s = _mm_add_pd(s, _mm_mul_pd(x8 , _mm_set1_pd(c8 )));
    s = _mm_add_pd(s, _mm_mul_pd(x10 , _mm_set1_pd(c10 )));
    _mm_store_pd(cosx+i, s);
  }
#else
  cos4_reference(cosx, x);
#endif
}

// ------------------------------------ End of Function ---------------------------------- //

double err(double* x, double* y, long N) {
  double error = 0;
  for (long i = 0; i < N; i++) {
    error = std::max(error, fabs(x[i]-y[i]));
  }
  return error;
}

int main() {
  Timer tt;
  long N = 1000000;
  double* x = (double*) aligned_malloc(N*sizeof(double));
  double* x_mod = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_ref = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_taylor = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_intrin = (double*) aligned_malloc(N*sizeof(double));
  double* cosx_taylor = (double*) aligned_malloc(N*sizeof(double));
  double* cosx_intrin = (double*) aligned_malloc(N*sizeof(double));
  int* flag = (int*) aligned_malloc(N*sizeof(int));


    for (long i=0; i<1; i++){
    x[i] =  (drand48()-0.5) * M_PI*2;   // [-pi, pi]
    sinx_ref[i] = 0;
    sinx_taylor[i] = 0;
    sinx_intrin[i] = 0;
    cosx_taylor[i] = 0;
    cosx_intrin[i] = 0;
    flag[i] = 10;
    x_mod[i] = 0;

  }
  for (long i=0; i<N ; i++){
    if(x[i] == 0 || x[i] == M_PI || x[i] == -1*M_PI){
      x_mod[i] = x[i];
      flag[i] = 0;
    }
    else if(x[i] >0 && x[i] <= M_PI/4){
      x_mod[i] = x[i];
      flag[i] = 0;
    }
    else if(x[i] >0 && x[i] > M_PI/4 && x[i] <= 3*M_PI/4){
      x_mod[i] = x[i] - M_PI/2;
      flag[i] = 1;
    }
    else if(x[i] >0 && x[i] > 3*M_PI/4 && x[i] < 4*M_PI/4){
      x_mod[i] = x[i] - M_PI;
      flag[i] = -2;
    }
    else if(x[i] <0 && x[i] >= (-1*M_PI/4)){
      x_mod[i] = x[i];
      flag[i] = 0;
    }
    else if(x[i] <0 && x[i] < (-1*M_PI/4) && x[i] >= (-3*M_PI/4)){
      x_mod[i] = x[i] + M_PI/2;
      flag[i] = -1;
    }
    else if(x[i] <0 && x[i] < (-3*M_PI/4) && x[i] > (-4*M_PI/4)){
      x_mod[i] = x[i] + M_PI;
      flag[i] = -2;
    }
  }


  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {   
    for (long i = 0; i < N; i+=4) {
      sin4_reference(sinx_ref+i, x+i);
    }
  }
  printf("Reference time: %6.4f\n", tt.toc());

  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_taylor(sinx_taylor+i, x_mod+i);
      cos4_taylor(cosx_taylor+i, x_mod+i);
    }
  }
    
    for (long i = 0; i < N; i++) {
      if(flag[i] == 0) sinx_taylor[i] = sinx_taylor[i];
      else if(flag[i] == -2) sinx_taylor[i] = sinx_taylor[i]*-1;
      else if (flag[i] == 1) sinx_taylor[i] = cosx_taylor[i];
      else if(flag[i] == -1) sinx_taylor[i] = cosx_taylor[i]*-1;
  }
  printf("Taylor time    %6.4f   Error: %e\n", tt.toc(), err(sinx_ref, sinx_taylor, N));

// -------------------------------------------------------------------------------------------------------//
  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) 
    {
      sin4_intrin(sinx_intrin+i, x_mod+i);
      cos4_intrin(cosx_intrin+i, x_mod+i);
    }

    for (long i = 0; i < N; i++) {
      if(flag[i] == 0) sinx_intrin[i] = sinx_intrin[i];
      else if(flag[i] == -2) sinx_intrin[i] = sinx_intrin[i]*-1;
      else if (flag[i] == 1) sinx_intrin[i] = cosx_intrin[i];
      else if(flag[i] == -1) sinx_intrin[i] = cosx_intrin[i]*-1;
    }
  }
  auto intrinTime = tt.toc();
  printf("Intrin time:    %6.4f      Error: %e\n", intrinTime, err(sinx_ref, sinx_intrin, N));

  aligned_free(x);
  aligned_free(sinx_ref);
  aligned_free(sinx_taylor);
  aligned_free(sinx_intrin);
  aligned_free(cosx_taylor);
  aligned_free(cosx_intrin);
  aligned_free(flag);

}