#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "immintrin.h"
#include <omp.h>
#include "kernel.h"
#include <time.h>

#define RUNS 1000

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

  int kernel_size  = 28 ;

  int k = 5;  
  int x_size = kernel_size * 8;      //number of data set points (arbitrary, can be changed)
  int a_size = 10;                   //number of test points (arbitrary, can be changed)
  
  int x_jump_size  = 28;  //x_jump_size is the number need to be jumped each time when calling kernel

  int p = 10;                               // total number of threads possible
  int total_thread_used = (a_size > p) ? p : a_size;   // total number of threads used

  //total_thread_used = 1; single thread
  FILE *fl, *fd, *ft, *fout;
  int dim =  10;
 

  fl = fopen("labels.txt","w");
  fd = fopen("datasets.txt","w");
  ft = fopen("testsets.txt","w");
  fout = fopen("outputlabel.txt","w");
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

    //initialize l
    for (int i = 0; i != x_size; ++i){
      double temp  = ((double) rand())/ ((double) RAND_MAX);
     if ( temp < 0.5)
      l[i] = 0;
    else
      l[i] = 1; 
    fprintf(fl,"%d\t",l[i]);
    }
    
    //initialize test_l 
    for (int i = 0; i != a_size; ++i){
      test_l[i] = 0;
    }

    // KNN implementation
    long sum = 0;       // count cycles consuming
    long t0 = rdtsc();
   

  for(int run=0;run<RUNS; run++){
    #pragma omp parallel for num_threads(total_thread_used)
    for (int j = 0; j < a_size*dim; j+=dim){
    int itt = 0;
    int temp1,temp2;
    int zero_count, one_count;
    double least_k_distance[k];
    bool least_k_label[k];
    int id = omp_get_thread_num();
    double *r = rr + id * x_size;      

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
          for(int j_i = 0; j_i<k; j_i++){
            if(least_k_distance[j_i] > max_distance){
              replace_index = j_i;
              max_distance = least_k_distance[j_i];
            }
          }
          if(replace_index > -1){
            least_k_distance[replace_index] = r[i];
            least_k_label[replace_index] = l[i];
          }
        }
      }

    // majority vote
      one_count = 0;
      for (int t = 0; t < k; ++t)
         // if(least_k[t] == 0)
		    one_count+=least_k_label[t];
      test_l[j/dim] = (one_count > k/2) ? true : false; 
 }
}
    // solution here 
    long t1 = rdtsc();
    
    sum += (t1 - t0);
    printf("............................................................\n");
    printf("instruction numbers for distance calculation %d\t, dimension: %d \n", a_size*x_size*3*dim, dim);
    printf("instruction numbers for find k least elements %d\t, dimension: %d \n", k*a_size*x_size + a_size*x_size, dim);
    printf("instruction numbers for label prediction %d\t, dimension: %d \n", (k-1)*a_size + a_size, dim);
    long total_instruction_nums = (RUNS*a_size*x_size*3*dim) + RUNS*(k*a_size*x_size + a_size*x_size) + (RUNS*((k-1)*a_size + a_size));
    printf("total instruction numbers %d\t, dimenssion: %d \n", total_instruction_nums/RUNS, dim);
    printf("performance (instructions/clock cycle): %lf\t, dimension: %d \n", total_instruction_nums/((double)(sum/(1.0))), dim);
    printf("time : %f\n",sum/(3.4*1000*1000*1000*RUNS));
    // printf("cycles consuming for distance calculation %lf\t, dimenssion: %d\n", (RUNS*a_size*x_size*3*dim)/((double)(sum/(1.0))), dim);
    // printf("cycles consuming for find k least elements %lf\t, dimenssion: %d\n", (RUNS*(k*a_size*x_size + a_size*x_size))/((double)(sum/(1.0))), dim);
    // printf("cycles consuming for find k least elements %lf\t, dimenssion: %d\n", (RUNS*((k-1)*a_size + a_size))/((double)(sum/(1.0))), dim);
    printf("............................................................\n");
    for(int i = 0; i< a_size; i++){
      fprintf(fout,"%d\t",test_l[i]);      
    }   
    
    free(x);
    free(a);
   free(rr);

  return 0;
}
