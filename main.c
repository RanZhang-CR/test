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
  double *rr;

  //double *c, *c_check;
  
  int starting_dimension = 10;  //  benchmark performance for 2D
  int ending_dimension  =  10;  //  benchmark performance for 5D
 // int dim = 2;
  int kernel_size  = 28 ;

  int k = 3;  
  int x_size = kernel_size * 4;     //number of data set points (arbitrary, can be changed)
  int a_size = 20;                   //number of test points (arbitrary, can be changed)
  
  int x_jump_size  = 28;  //x_jump_size is the number need to be jumped each time when calling kernel

  int p = 10;                               // total number of threads possible
  int total_thread_used = (a_size > p) ? p : a_size;   // total number of threads used

  FILE *fl, *fd, *ft, *fout;
  fl = fopen("labels.txt","w");
  fd = fopen("datasets.txt","w");
  ft = fopen("testsets.txt","w");
  fout = fopen("outputlabel.txt","w");
  
  for (int dim = starting_dimension ; dim <= ending_dimension ;dim++){

    posix_memalign((void**) &x, 64, x_size * dim * sizeof(double));
    posix_memalign((void**) &a, 64, a_size * dim * sizeof(double));
    posix_memalign((void**) &rr, 64, x_size * total_thread_used * sizeof(double));     
    posix_memalign((void**) &l, 64, x_size * sizeof(bool));
    posix_memalign((void**) &test_l, 64, a_size * sizeof(bool));

    //initialize data set 

    for (int i = 0; i != x_size * dim; ++i){
      x[i] = ((double) rand())/ ((double) RAND_MAX);
      fprintf(fd,"%f\t",x[i]);
    }
    
    //initialize test points
    for (int i = 0; i != a_size * dim; ++i){
      a[i] = ((double) rand())/ ((double) RAND_MAX);
      fprintf(ft,"%f\t",a[i]);
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

      // write the labels into "labels.txt"
      fprintf(fl,"%d\t",l[i]);
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

for(int run=0;run<RUNS; run++){
    #pragma omp parallel for num_threads(total_thread_used)
    for (int j = 0; j < a_size*dim; j+=dim){
    int itt = 0;
    int temp1,temp2;
    int zero_count, one_count;
    double least_k_distance[k];
    bool least_k_label[k];
    // posix_memalign((void**) &least_k, 64, k * sizeof(bool));
    int id = omp_get_thread_num();
    double *r = rr + id * x_size;      
    // double *r = &rr[id * x_size];
    // posix_memalign((void**) &r, 64, x_size * sizeof(double)); 

    for (int i = 0; i != x_size; ++i){
      r[i] = 0;
    }

      // get Euclidiant distance for each point 
      //loop for each point in the dataset
      for (int i = 0; i < x_size*dim; i+=kernel_size*dim){	
          kernel(x+i, a+j, r + itt*kernel_size, dim);
          itt++;
      }

      // find the least k values
      for(int i = 0; i < x_size; i++){
        // copy the first k elements
        if(i<k){
          least_k_distance[i] = r[i];
          least_k_label[i] = l[i]; 
        }
        // begin from kth element, compare it with the elements in least_k_distances,
        // if smaller than the maximum element in least_k_distances, replaces it
        else{
          int replace_index = -1;
          double max_distance = r[i];
          for(int j = 0; j<k; j++){
            if(least_k_distance[j] > max_distance){
              replace_index = j;
              max_distance = least_k_distance[j];
            }
          }
          if(replace_index > -1){
            least_k_distance[replace_index] = r[i];
            least_k_label[replace_index] = l[i];
          }
        }
      }

    // majority vote
      zero_count = 0;
      for (int t = 0; t < k; ++t)
         // if(least_k[t] == 0)
		    zero_count+=(least_k_label[t] == 0);
      test_l[j/dim] = (zero_count > k/2) ? false : true;
    // free(r);   
 }
}
    // solution here 
    int t1 = rdtsc();
    sum += (t1 - t0);

    printf("instruction numbers for distance calculation %d\t, dimenssion: %d\n", a_size*x_size*3*dim, dim);
    printf("instruction numbers for find k least elements %d\t, dimenssion: %d\n", k*a_size*x_size + a_size*x_size, dim);
    printf("instruction numbers for label prediction %d\t, dimenssion: %d\n", (k-1)*a_size + a_size, dim);
    int total_instruction_nums = (RUNS*a_size*x_size*3*dim) + RUNS*(k*a_size*x_size + a_size*x_size) + (RUNS*((k-1)*a_size + a_size));
    printf("total instruction numbers %d\t, dimenssion: %d\n", total_instruction_nums/RUNS, dim);
    printf("performance (instructions/clock cycle): %lf\t, dimenssion: %d\n", total_instruction_nums/((double)(sum/(1.0))), dim);

    // printf("cycles consuming for distance calculation %lf\t, dimenssion: %d\n", (RUNS*a_size*x_size*3*dim)/((double)(sum/(1.0))), dim);
    // printf("cycles consuming for find k least elements %lf\t, dimenssion: %d\n", (RUNS*(k*a_size*x_size + a_size*x_size))/((double)(sum/(1.0))), dim);
    // printf("cycles consuming for find k least elements %lf\t, dimenssion: %d\n", (RUNS*((k-1)*a_size + a_size))/((double)(sum/(1.0))), dim);
    for(int i = 0; i< a_size; i++){
      fprintf(fout,"%d\t",test_l[i]);      
    }
    
    free(x);
    free(a);
   free(rr);
  }

  return 0;
}
