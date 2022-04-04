#include <math.h>
#include <stdio.h>
#include "utils.h"
#include "intrin-wrapper.h"

// coefficients in the Taylor series expansion of sin(x)
static constexpr double c3  = -1/(((double)2)*3);
static constexpr double c5  =  1/(((double)2)*3*4*5);
static constexpr double c7  = -1/(((double)2)*3*4*5*6*7);
static constexpr double c9  =  1/(((double)2)*3*4*5*6*7*8*9);
static constexpr double c11 = -1/(((double)2)*3*4*5*6*7*8*9*10*11);
// sin(x) = x + c3*x^3 + c5*x^5 + c7*x^7 + x9*x^9 + c11*x^11

// coefficients in the Taylor series expansion of cos(x)
static constexpr double c2  = -1/(((double)2)*3);
static constexpr double c4  =  1/(((double)2)*3*4*5);
static constexpr double c6  = -1/(((double)2)*3*4*5*6*7);
static constexpr double c8  =  1/(((double)2)*3*4*5*6*7*8*9);
static constexpr double c10 = -1/(((double)2)*3*4*5*6*7*8*9*10*11);
static constexpr double c12 =  1/(((double)2)*3*4*5*6*7*8*9*10*11*12);
// cos(x) = 1 + c2*x^2 + c4*x^4 + c6*x^6 + c8*x^6 + c10*x^10 + c12*x^12

void sin4_reference(double* sinx, const double* x) {
  for (long i = 0; i < 4; i++) sinx[i] = sin(x[i]);
}

void sin4_taylor(double* sinx, const double* x, const int* flag_copy) {
  for (int i = 0; i < 4; i++) {
    double x1  = x[i];
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
    if (flag_copy[i] == -2) sinx[i] = -1*s;
    else sinx[i] = s;
  }
}

void cos4_taylor(double* cosx, const double* x, const int* flag_copy) {
  for (int i = 0; i < 4; i++) {
    double x0 = 1;
    double x1  = x[i];
    double x2  = x1 * x1;
    double x4  = x2 * x2;
    double x6  = x4 * x2;
    double x8  = x6 * x2;
    double x10 = x8 * x2;
    double x12 = x10 * x2;

    double s = x0;
    s += x2  * c2;
    s += x4  * c4;
    s += x6  * c6;
    s += x8  * c8;
    s += x10 * c10;
    s += x12 * c12;
    if (flag_copy[i] == -1) cosx[i] = -1*s;
    else cosx[i] = s;
  }
}

double err(double* x, double* y, double* z, long N, int* flag_copy) {
  double error = 0;
  for (long i = 0; i < N; i++) {
    if (flag_copy[i] == 0 || flag_copy[i] == -2) error = std::max(error, fabs(x[i]-y[i]));
    else if (flag_copy[i] == 1 || flag_copy[i] == -1) error = std::max(error, fabs(x[i]-z[i]));
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

  for (long i = 0; i < N; i++) {
    x[i] = (drand48()-0.5) * M_PI/2; 
    //x_mod[i] = x[i];
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
      if(flag[i] == 0 || flag[i] == -2) sin4_taylor(sinx_taylor+i, x_mod+i, flag+i);
      else if(flag[i] == 1 || flag[i] == -1)    cos4_taylor(cosx_taylor+i, x_mod+i, flag+i);
    }
  }
  printf("Taylor time    %6.4f   Error: %e\n", tt.toc(), err(sinx_ref, sinx_taylor, cosx_taylor, N, flag));

  aligned_free(x);
  aligned_free(sinx_ref);
  aligned_free(sinx_taylor);
  aligned_free(sinx_intrin);
  aligned_free(cosx_taylor);
  aligned_free(cosx_intrin);
  aligned_free(flag);

}
