#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "immintrin.h"

#include "kernel.h"

#define RUNS 5000


int main(){
  //data sets    
  //[x1,x2,x3,x4...x28,y1,y2,y3,y4....y28, ..... , final_dimension1,final_dimension2,...final_dimension28]
  double *x;   

  //test points
  //[a1,b1,c1, ..., final_dimension1, a2, b2,c2, ... final_dimension2, ..., ]    
  double *a;      

  bool *l;      // training labels for x

  bool *test_l  // finding labels for test point      


  // r[i] is the distance between one training point ans one test point
  double *r;

  //double *c, *c_check;
  
  int starting_dimension = 2;  //  benchmark performance for 2D
  int ending_dimension  =  5;  //  benchmark performance for 5D

  int RUNS = 100;    // can be changed


  int kernel_size  = 28 ;


  int x_size = kernel_size * 10;     //number of data set points (arbitrary, can be changed)
  int a_size = 20;                   //number of test points (arbitrary, can be changed)
  
  int x_jump_size  = 28;  //x_jump_size is the number need to be jumped each time when calling kernel

 

  for (int k = starting_dimension ; k <= ending_dimension ;k++){

    posix_memalign((void**) &x, 64, x_size * k * sizeof(double));
    posix_memalign((void**) &a, 64, a_size * k * sizeof(double));
    posix_memalign((void**) &r, 64, a_size * x_size * sizeof(double));
    posix_memalign((void**) &l, 64, x_size * sizeof(bool));
    posix_memalign((void**) &test_l, 64, a_size * sizeof(bool));
    

    //initialize data set 
    for (int i = 0; i != x_size * k; ++i){
      x[i] = ((double) rand())/ ((double) RAND_MAX);
    }
    //initialize test points
    for (int i = 0; i != a_size * k; ++i){
      a[i] = ((double) rand())/ ((double) RAND_MAX);
    }

    //initialize r 
    for (int i = 0; i != a_size * x_size; ++i){
      r[i] = ((double) rand())/ ((double) RAND_MAX);
    }


    //initialize l
    for (int i = 0; i != x_size; ++i){
      double temp  = ((double) rand())/ ((double) RAND_MAX);
     if ( temp < 0.5)
      l[i] = 0;
    else
      l[i] = 1; 
    }
    }    
    //initialize test_l 
    for (int i = 0; i != a_size; ++i){
      test_l[i] = 0;
    }


    // KNN implementation
    sum = 0;       // count cycles consuming
    for (int runs = 0; runs != RUNS; ++runs){
      t0 = rdtsc();
      itt = 0;
      for (int j = 0; j != a_size*k; j+=k){
        for (int i = 0; i != x_size*k; i+=kernel_size*k){	
            kernel(x+i, a+j, r + itt*kernel_size, k);
            itt++;    
        }
      }
      t1 = rdtsc();
      sum += (t1 - t0);
    }
    /*
    int correct = 1;
    for (int i = 0; i != m * n; ++i)
      correct &= (fabs(c[i] - c_check[i]) < 1e-12);
    printf(" %lf\t%d\n", (2.0*mc*n*k)/((double)(sum/(1.0*RUNS))), correct);
    */
    free(x);
    free(a);
    free(r);
    free(l);
  }

  
  return 0;
}
