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


  int x_size = kernel_size * 20;     //number of data set points (arbitrary, can be changed)
  int a_size = 20;                   //number of test points (arbitrary, can be changed)
  
  int x_jump_size  = 28;  //x_jump_size is the number need to be jumped each time when calling kernel

 
  
  for ( int k = starting_dimension ; k <= ending_dimension ;k++){

    posix_memalign((void**) &x, 64, x_size * k * sizeof(double));
    posix_memalign((void**) &a, 64, a_size * k * sizeof(double));
    posix_memalign((void**) &r, 64, a_size * x_size * sizeof(double));
    posix_memalign((void**) &l, 64, x_size * sizeof(bool));
    posix_memalign((void**) &test_l, 64, a_size * sizeof(bool));
    

    //initialize data set 
    for (int i = 0; i != x_size * k; ++i){
      x[i] = ((double) rand())/ ((double) RAND_MAX);
      //printf("%lf,",x[i]);
      if(i==x_size-1){
        //printf("y:\n");
      }
    }
    //printf("\n a: \n");
    //initialize test points
    for (int i = 0; i != a_size * k; ++i){
      a[i] = ((double) rand())/ ((double) RAND_MAX);
      //printf("%lf\t",a[i]);
    }
   // printf("\n");
    //initialize r 
    for (int i = 0; i != a_size * x_size; ++i){
      r[i] = 0;//((double) rand())/ ((double) RAND_MAX);
    }

   // printf("l : \n");
    //initialize l
    for (int i = 0; i != x_size; ++i){
      double temp  = ((double) rand())/ ((double) RAND_MAX);
     if ( temp < 0.5)
      l[i] = 0;
    else
      l[i] = 1; 
     // printf("%d\t",l[i]);
    }
   // printf("\n");
  
    //initialize test_l 
    for (int i = 0; i != a_size; ++i){
      test_l[i] = 0;
    }


    // KNN implementation
    int sum = 0;       // count cycles consuming
    for (int runs = 0; runs != RUNS; ++runs){
      int t0 = rdtsc();
      int itt = 0;
      for (int j = 0; j != a_size*k; j+=k){
        for (int i = 0; i != x_size*k; i+=kernel_size*k){	
            //printf("i=%d j=%d itt=%d k=%d \n",i,j,itt*kernel_size,k);
            kernel(x+i, a+j, r + itt*kernel_size, k);
            itt++;    
        }
      }
      
      // double sumr = 0;
      // for(int m=0;m<28;m++){
      //   sumr += r[m];
      // }
      // printf("%lf",sumr);
      
      int t1 = rdtsc();
      sum += (t1 - t0);
    }
    
    int correct = 1;
    // for (int i = 0; i != m * n; ++i)
    //   correct &= (fabs(c[i] - c_check[i]) < 1e-12);
    printf("cycles consuming %lf\t%d, dimenssion: %d\n", (a_size*x_size*5*k)/((double)(sum/(1.0*RUNS))), correct, k);
    
    free(x);
    free(a);
    free(r);
  }

  
  return 0;
}
