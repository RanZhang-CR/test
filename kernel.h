#include "immintrin.h"

void inline kernel(
    double* x;
    double* a;
    double* r;

    int D;    // dimension size
){
    __m256d smd8 = _mm256_setzero_pd();
    __m256d smd9 = _mm256_setzero_pd();
    __m256d smd10 = _mm256_setzero_pd();
    __m256d smd11 = _mm256_setzero_pd();
    __m256d smd12 = _mm256_setzero_pd();
    __m256d smd13 = _mm256_setzero_pd();
    __m256d smd14 = _mm256_setzero_pd();

    for(int i=0; i < D; ++i){
        // load data
        __m256d smd1 = _mm256_loadu_pd(x + i*28);
        __m256d smd2 = _mm256_loadu_pd(x+4+ i*28);
        __m256d smd3 = _mm256_loadu_pd(x+8+ i*28);
        __m256d smd4 = _mm256_loadu_pd(x+12+ i*28);
        __m256d smd5 = _mm256_loadu_pd(x+16+ i*28);
        __m256d smd6 = _mm256_loadu_pd(x+20+ i*28);
        __m256d smd7 = _mm256_loadu_pd(x+24+ i*28);

        __m256d smd0 = _mm256_broadcast_sd(a + i);

        // __m256d smd8 = _mm256_loadu_pd(r);
        // __m256d smd9 = _mm256_loadu_pd(r+4);
        // __m256d smd10 = _mm256_loadu_pd(r+8);
        // __m256d smd11 = _mm256_loadu_pd(r+12);
        // __m256d smd12 = _mm256_loadu_pd(r+16);
        // __m256d smd13 = _mm256_loadu_pd(r+20);
        // __m256d smd14 = _mm256_loadu_pd(r+24);


        // subtraction part
        smd1 = _mm256_sub_pd(smd0, smd1);
        smd2 = _mm256_sub_pd(smd0, smd2);
        smd3 = _mm256_sub_pd(smd0, smd3);
        smd4 = _mm256_sub_pd(smd0, smd4);
        smd5 = _mm256_sub_pd(smd0, smd5);
        smd6 = _mm256_sub_pd(smd0, smd6);
        smd7 = _mm256_sub_pd(smd0, smd7);

        // fma part
        smd8 = _mm256_fmadd_pd(smd1, smd1, smd8);
        smd9 = _mm256_fmadd_pd(smd2, smd2, smd9);
        smd10 = _mm256_fmadd_pd(smd3, smd3, smd10);
        smd11 = _mm256_fmadd_pd(smd4, smd4, smd11);
        smd12 = _mm256_fmadd_pd(smd5, smd5, smd12);
        smd13 = _mm256_fmadd_pd(smd6, smd6, smd13);
        smd14 = _mm256_fmadd_pd(smd7, smd7, smd14);
    }


    // store 28 distances
    _mm256_storeu_pd(r, smd8);
    _mm256_storeu_pd(r+4, smd9);
    _mm256_storeu_pd(r+8, smd10);
    _mm256_storeu_pd(r+12, smd11);
    _mm256_storeu_pd(r+16, smd12);
    _mm256_storeu_pd(r+20, smd13);
    _mm256_storeu_pd(r+24, smd14);
}
