#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "immintrin.h"

#include "kernel.h"

#define RUNS 2000

//timing routine for reading the time stamp counter
static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

int main(){
  //data sets    
  //[x1,x2,x3,x4...x28,y1,y2,y3,y4....y28, ..... , final_dimension1,final_dimension2,...final_dimension28]
  double *x;   
 
  double *leastKvalue;
  //test points
  //[a1,b1,c1, ..., final_dimension1, a2, b2,c2, ... final_dimension2, ..., ]    
  double *a;      

  bool *l;      // training labels for x

  bool *test_l;  // finding labels for test point      


  // r[i] is the distance between one training point ans one test point
  double *r;

  //double *c, *c_check;
  
  int starting_dimension = 1;  //  benchmark performance for 2D
  int ending_dimension  =  10;  //  benchmark performance for 5D

  int kernel_size  = 28 ;

  int k = 3;  
  int x_size = kernel_size * 40;     //number of data set points (arbitrary, can be changed)
  int a_size = 20;                   //number of test points (arbitrary, can be changed)
  
  int x_jump_size  = 28;  //x_jump_size is the number need to be jumped each time when calling kernel

 
  
  for ( int dim = starting_dimension ; dim <= ending_dimension ;dim++){

    posix_memalign((void**) &x, 64, x_size * dim * sizeof(double));
    posix_memalign((void**) &a, 64, a_size * dim * sizeof(double));
    posix_memalign((void**) &r, 64, a_size * x_size * sizeof(double));
    posix_memalign((void**) &l, 64, x_size * sizeof(bool));
    posix_memalign((void**) &test_l, 64, a_size * sizeof(bool));
    posix_memalign((void**) &leastKvalue, 64, k * sizeof(double));

    //initialize data set 
    for (int i = 0; i != x_size * dim; ++i){
      x[i] = ((double) rand())/ ((double) RAND_MAX);
    }

    //initialize test points
    for (int i = 0; i != a_size * dim; ++i){
      a[i] = ((double) rand())/ ((double) RAND_MAX);
    }

    //initialize r 
    for (int i = 0; i != a_size * x_size; ++i){
      r[i] = 0;
    }

    //initialize l
    for (int i = 0; i != x_size; ++i){
      double temp  = ((double) rand())/ ((double) RAND_MAX);
     if ( temp < 0.5)
      l[i] = 0;
    else
      l[i] = 1; 
    }
  
    //initialize test_l 
    for (int i = 0; i != a_size; ++i){
      test_l[i] = 0;
    }

    // KNN implementation
    int sum = 0;       // count cycles consuming
    int t0 = rdtsc();
    int itt = 0;
    for (int j = 0; j != a_size*dim; j+=dim){
      for (int i = 0; i != x_size*dim; i+=kernel_size*dim){	
          kernel(x+i, a+j, r + itt*kernel_size, dim,k,leastKvalue);
          itt++;    
      }
    }
    // solution here 
    int t1 = rdtsc();
    sum += (t1 - t0);
  
    printf("cycles consuming %lf\t, dimenssion: %d\n", (a_size*x_size*3*dim)/((double)(sum/(1.0))), dim);
    
    free(x);
    free(a);
    free(r);
  }

  return 0;
}
