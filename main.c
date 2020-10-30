#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "immintrin.h"

#include "kernel.h"

#define RUNS 5000


int main(){
  //data sets    
  //[x1,x2,x3,x4...x28,y1,y2,y3,y4....y28, ..... , final_dimension1,final_dimension2,...final_dimension28]
  double *a;   

  //test points
  //[a1,b1,c1, ..., final_dimension1, a2, b2,c2, ... final_dimension2, ..., ]    
  double *b;      

  double *l;      //labels

  // distances without squre root
  double *r;

  //double *c, *c_check;


  // TODO on OCT 31st
  unsigned long long t0, t1, sum;

  //CHANGE THESE
  int mc = 96;  //mc is the number of rows of A
  int kc = 256;  //kc is the number of columns of A

  //DO NOT CHANGE THESE
  int m  = 6;  //m is the number of rows of the kernel
  int n  = 8;  //n is the number of columns of the kernel

  /*
    Assume the following
        - A is stored in column major order
        - B is stored in row major order
        - C must be written out in row major order
  */

  for (int k = kc/2; k <= 2*kc; k *= 2){

    posix_memalign((void**) &a, 64, mc * k * sizeof(double));
    posix_memalign((void**) &b, 64, n * k * sizeof(double));
    posix_memalign((void**) &c, 64, mc * n * sizeof(double));
    posix_memalign((void**) &c_check, 64, mc * n * sizeof(double));

    //initialize A
    for (int i = 0; i != k * mc; ++i){
      a[i] = ((double) rand())/ ((double) RAND_MAX);
    }
    //initialize B
    for (int i = 0; i != k * n; ++i){
      b[i] = ((double) rand())/ ((double) RAND_MAX);
    }
    //initialize C
    for (int i = 0; i != mc * n; ++i){
      c[i] = 0.0;
      c_check[i] = 0.0;
    }

    printf("%d\t %d\t %d\t", mc, n, k);

    sum = 0;
    for (int runs = 0; runs != RUNS; ++runs){
      t0 = rdtsc();
      for (int i = 0; i != mc / m; ++i){	
	naive(6, 8, k, a + i*m*k, b, c_check + i*6*8);      
      }
      t1 = rdtsc();
      sum += (t1 - t0);
    }
    printf(" %lf\t", (2.0*mc*n*k)/(sum/(1.0*RUNS)));  



    sum = 0;
    for (int runs = 0; runs != RUNS; ++runs){
      t0 = rdtsc();
      for (int i = 0; i != mc / m; ++i){	
	kernel(k, a + i*m*k, b, c + i*6*8);      
      }
      t1 = rdtsc();
      sum += (t1 - t0);
    }

    int correct = 1;
    for (int i = 0; i != m * n; ++i)
      correct &= (fabs(c[i] - c_check[i]) < 1e-12);
    printf(" %lf\t%d\n", (2.0*mc*n*k)/((double)(sum/(1.0*RUNS))), correct);

    free(a);
    free(b);
    free(c);
    free(c_check);
  }

  
  return 0;
}
