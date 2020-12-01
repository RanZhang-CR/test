#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "immintrin.h"
#include <omp.h>
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
 // int dim = 2;
  int kernel_size  = 28 ;

  int k = 3;  
  int x_size = kernel_size * 4;     //number of data set points (arbitrary, can be changed)
  int a_size = 20;                   //number of test points (arbitrary, can be changed)
  
  int x_jump_size  = 28;  //x_jump_size is the number need to be jumped each time when calling kernel

 
  
  for (int dim = starting_dimension ; dim <= ending_dimension ;dim++){

    posix_memalign((void**) &x, 64, x_size * dim * sizeof(double));
    posix_memalign((void**) &a, 64, a_size * dim * sizeof(double));
   // posix_memalign((void**) &r, 64, x_size * sizeof(double));
    posix_memalign((void**) &l, 64, x_size * sizeof(bool));
    posix_memalign((void**) &test_l, 64, a_size * sizeof(bool));

    //initialize data set 

    for (int i = 0; i != x_size * dim; ++i){
      x[i] = ((double) rand())/ ((double) RAND_MAX);
    }
    
    //initialize test points
    for (int i = 0; i != a_size * dim; ++i){
      a[i] = ((double) rand())/ ((double) RAND_MAX);
    }

    //initialize r 
   /* for (int i = 0; i != x_size; ++i){
      r[i] = 0;
    }
*/
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
   // int itt = 0;
   // int temp1,temp2;
   // int zero_count, one_count;
    //loop for each test point
   // printf("entering parallel");
   // fflush(0);
    #pragma omp parallel for
    for (int j = 0; j < a_size*dim; j+=dim){
     // itt = 0;
    int itt = 0;
    int itter = 0;
    int temp1,temp2;
    int zero_count, one_count;
    double least_k[k];
    posix_memalign((void**) &least_k, 64, k * sizeof(bool));
    double *r;
    posix_memalign((void**) &r, 64, x_size * sizeof(double)); 
    for (int i = 0; i != x_size; ++i){
      r[i] = 0;
    }
      //loop for each point in the dataset
      for (int i = 0; i < x_size*dim; i+=kernel_size*dim){	
          kernel(x+i, a+j, r + itt*kernel_size, dim);
          itt++;
      }
      // find the least k values
      // selection sort 
      for(int sort_x = 0; sort_x < k; sort_x++){
        int min = r[sort_x];
	int pos = sort_x;
        for(int sort_y = sort_x+1; sort_y < x_size; sort_y++)
             if(min > r[sort_y]){
                 min = r[sort_y];
                 pos = sort_y;
              }
        temp1 = r[sort_x];
        r[sort_x] = r[pos];
        r[pos] = temp1;

       least_k[itter] = l[pos];
       itter++;
      }
    // majority vote
      one_count = 0;
      zero_count = 0;
      for (int t = 0; t < k; ++t)
         // if(least_k[t] == 0)
		zero_count+=(least_k[t] == 0);
//	  else
//		one_count++;  
      test_l[j/dim] = (zero_count > k/2) ? false : true;
    free(r);   
 }
    
    // solution here 
    int t1 = rdtsc();
    sum += (t1 - t0);
  
    printf("cycles consuming %lf\t, dimenssion: %d\n", (a_size*x_size*3*dim)/((double)(sum/(1.0))), dim);
    
    free(x);
    free(a);
 //   free(r);
  }

  return 0;
}
