/*
Parallel version
Corresponding Algorithm 9
*/

#include <stdio.h>
#include <stdint.h>
#include "params.h"
#include "poly.h"
#include "mult.h"
#include <immintrin.h>
void prepare_s1_table(uint64_t s_table[2*N], const polyvecl *s1)
{
	uint32_t k,j;
    uint64_t temp;
	uint64_t mask_s = 0x8040201008;
	for(k=0; k<N; k++){
        //s_table[k+N] = 0;

        for(j=0; j<L; j++)
		{
			temp = (uint64_t)(ETA + s1->vec[j].coeffs[k]);
			s_table[k+N] = (s_table[k+N]<<9) | (temp);
		}
        s_table[k] = mask_s - s_table[k+N];

    }

}
void prepare_s1_table_32(uint32_t s11_table[2*N], uint32_t s12_table[2*N],const polyvecl *s1)
{
	uint32_t k,j;
    uint64_t temp;
	uint32_t mask_s11 = 0x201008;
    uint32_t mask_s12 = 0x1008;
	for(k=0; k<N; k++){

        for(j=0; j<3; j++)
		{
			temp = (uint32_t)(ETA + s1->vec[j].coeffs[k]);
			s11_table[k+N] = (s11_table[k+N]<<9) | (temp);
		}
        for(j=3; j<L; j++)
		{
			temp = (uint32_t)(ETA + s1->vec[j].coeffs[k]);
			s12_table[k+N] = (s12_table[k+N]<<9) | (temp);
		}
        s11_table[k] = mask_s11 - s11_table[k+N];
        s12_table[k] = mask_s12 - s12_table[k+N];
    }
}
void prepare_s1_table_32_avx2(uint32_t s11_table[2*N], uint32_t s12_table[2*N],const polyvecl *s1)
{
	uint32_t k,j;
    __m256i s1_coeffs_vec, s11_table_vec, s12_table_vec;
    const __m256i mask_s11 = _mm256_set1_epi32(2101256);
    const __m256i mask_s12 = _mm256_set1_epi32(4104);
    const __m256i eta = _mm256_set1_epi32(ETA);
    for(k=0; k<N; k+=8){
        for(j=0; j<3; j++)
		{
            s1_coeffs_vec = _mm256_load_si256((__m256i*)&s1->vec[j].coeffs[k]);
            s1_coeffs_vec = _mm256_add_epi32(eta, s1_coeffs_vec);
            s11_table_vec = _mm256_slli_epi32(s11_table_vec,9);
            s11_table_vec = _mm256_or_si256(s11_table_vec, s1_coeffs_vec);
		}
        for(j=3; j<L; j++)
		{
			s1_coeffs_vec = _mm256_load_si256((__m256i*)&s1->vec[j].coeffs[k]);
            s1_coeffs_vec = _mm256_add_epi32(eta,s1_coeffs_vec);
            s12_table_vec = _mm256_slli_epi32(s12_table_vec,9);
            s12_table_vec = _mm256_or_si256(s12_table_vec, s1_coeffs_vec);
		}
        _mm256_store_si256((__m256i*)&s11_table[k+N],s11_table_vec);
        s11_table_vec = _mm256_sub_epi32(mask_s11, s11_table_vec);
        _mm256_store_si256((__m256i*)&s11_table[k],s11_table_vec);
        _mm256_store_si256((__m256i*)&s12_table[k+N],s12_table_vec);
        s12_table_vec = _mm256_sub_epi32(mask_s12, s12_table_vec);
        _mm256_store_si256((__m256i*)&s12_table[k],s12_table_vec);
    }
}
int evaluate_cs1_earlycheck_32(polyvecl *z, polyvecl *y, const poly *c, const uint32_t s11_table[2*N], const uint32_t s12_table[2*N], int32_t B)
{
    int i, j;
    uint32_t w[N] = {0};
    uint32_t w2[N] = {0};
    uint32_t mask_s11 = 0x201008;
    uint32_t mask_s12 = 0x1008;
    for(i = 0; i < N; i ++)
    {
        if(c->coeffs[i] == 1)
        {
            for(j = 0; j < N; j ++)
            {
                w[j] = w[j] + s11_table[j-i+N];
                w2[j] = w2[j] + s12_table[j-i+N];
            }
        }
        if(c->coeffs[i] == -1)
        {
            for(j = 0; j < N; j ++)
            {
                w[j] = w[j] + (mask_s11 - s11_table[j-i+N]);
                w2[j] = w2[j] + (mask_s12 - s12_table[j-i+N]);
            }
        }
    }
    uint32_t temp, temp2;
    for(i=0; i<N; i++)
	{
        temp = w2[i];

        for(j=0; j<2 ; j++)
		{
			z->vec[L-1-j].coeffs[i] = ((int32_t)(temp & 0x1FF)-TAU*ETA);//(t mod M) - rU (mod q)
			temp >>= 9;//t = t/M
            z->vec[L-1-j].coeffs[i] = y->vec[L-1-j].coeffs[i] + z->vec[L-1-j].coeffs[i];
            if(z->vec[L-1-j].coeffs[i] >= B || z->vec[L-1-j].coeffs[i] <= -B) {
                return 1;
            }
		}
        temp2 = w[i];
        for(j=2;j < L; j++)
        {
            z->vec[L-1-j].coeffs[i] = ((int32_t)(temp2 & 0x1FF)-TAU*ETA);//(t mod M) - rU (mod q)
			temp2 >>= 9;//t = t/M
            z->vec[L-1-j].coeffs[i] = y->vec[L-1-j].coeffs[i] + z->vec[L-1-j].coeffs[i];
            if(z->vec[L-1-j].coeffs[i] >= B || z->vec[L-1-j].coeffs[i] <= -B) {
                return 1;
            }
        }
	}
    return 0;
}
int evaluate_cs1_earlycheck_32_avx2(polyvecl *z, polyvecl *y, const poly *c, const uint32_t s11_table[2*N], const uint32_t s12_table[2*N], int32_t B)
{
    int i, j;
    __attribute__((aligned(32)))  uint32_t w[N] = {0};
    __attribute__((aligned(32)))  uint32_t w2[N] = {0};
    for(i = 0; i < N; i ++)
    {
        if(c->coeffs[i] == 1){

            add_asm(w, &s11_table[N - i]);
            add_asm(w2, &s12_table[N - i]);
        }
        else if(c->coeffs[i] == -1){

            addmask11_asm(w, &s11_table[N - i]);
            addmask12_asm(w2, &s12_table[N - i]);
        }
    }
    //recover
     __m256i w_vec, w2_vec, z1_vec, z2_vec, y_vec, zabs_vec, cmp1_vec, cmpeq_vec;
    int cmp1_res, cmpeq_res;
    const __m256i mask_vec = _mm256_set1_epi32(0x1FF);
    const __m256i taueta_vec = _mm256_set1_epi32(TAU*ETA);
    const __m256i boundB_vec = _mm256_set1_epi32(B);
    for(i=0; i<N; i+=8)
	{
        w_vec =_mm256_loadu_si256((__m256i*)&w[i]);
        w2_vec =_mm256_loadu_si256((__m256i*)&w2[i]);

        for(j=0; j<2 ; j++)
		{
			z2_vec =_mm256_and_si256(w2_vec, mask_vec);
            z2_vec =_mm256_sub_epi32(z2_vec, taueta_vec);
            y_vec = _mm256_loadu_si256((__m256i*)&y->vec[L-1-j].coeffs[i]);
            z2_vec = _mm256_add_epi32(z2_vec, y_vec);
            zabs_vec = _mm256_abs_epi32(z2_vec);
            cmp1_vec = _mm256_sub_epi32(boundB_vec, zabs_vec);
            cmp1_res = _mm256_movemask_ps((__m256)cmp1_vec);
            cmpeq_vec = _mm256_cmpeq_epi32(boundB_vec, zabs_vec);
            cmpeq_res = _mm256_movemask_ps((__m256)cmpeq_vec);
            if(cmp1_res || cmpeq_res)
            {
                return 1;
            }
            w2_vec = _mm256_srli_epi32(w2_vec, 9);
            _mm256_storeu_si256((__m256i*)&z->vec[L-1-j].coeffs[i],z2_vec);
		}
        for(j=2;j < L; j++)
        {
            z1_vec =_mm256_and_si256(w_vec, mask_vec);
            z1_vec =_mm256_sub_epi32(z1_vec, taueta_vec);
            y_vec = _mm256_loadu_si256((__m256i*)&y->vec[L-1-j].coeffs[i]);
            z1_vec = _mm256_add_epi32(z1_vec, y_vec);
            zabs_vec = _mm256_abs_epi32(z1_vec);
            cmp1_vec = _mm256_sub_epi32(boundB_vec, zabs_vec);
            cmp1_res = _mm256_movemask_ps((__m256)cmp1_vec);
            cmpeq_vec = _mm256_cmpeq_epi32(boundB_vec, zabs_vec);
            cmpeq_res = _mm256_movemask_ps((__m256)cmpeq_vec);
            if(cmp1_res || cmpeq_res)
            {
                return 1;
            }
            w_vec = _mm256_srli_epi32(w_vec, 9);
            _mm256_storeu_si256((__m256i*)&z->vec[L-1-j].coeffs[i],z1_vec); 
        }
	}
    return 0;
}
int evaluate_cs1_earlycheck(polyvecl *z, polyvecl *y, const poly *c, const uint64_t s_table[2*N], int32_t B)
{
    int i, j;
    uint64_t w[N] = {0};
    uint64_t mask_s = 0x8040201008;
    for(i = 0; i < N; i ++)
    {
        if(c->coeffs[i] == 1)
        {
            for(j = 0; j < N; j ++)
            {
                w[j] = w[j] + s_table[j-i+N];
            }
        }
        if(c->coeffs[i] == -1)
        {
            for(j = 0; j < N; j ++)
            {
                w[j] = w[j] + (mask_s - s_table[j-i+N]);
            }
        }
    }
    uint64_t temp;
    // int32_t t;
    for(i=0; i<N; i++)
	{
        temp = w[i];

        for(j=0; j<L; j++)
		{
			z->vec[L-1-j].coeffs[i] = ((int32_t)(temp & 0x1FF)-TAU*ETA);//(t mod M) - rU (mod q)
			temp >>= 9;//t = t/M
            z->vec[L-1-j].coeffs[i] = y->vec[L-1-j].coeffs[i] + z->vec[L-1-j].coeffs[i];
            // t = z->vec[L-1-j].coeffs[i] >> 31;
            // t = z->vec[L-1-j].coeffs[i] - (t & 2*z->vec[L-1-j].coeffs[i]);
            if(z->vec[L-1-j].coeffs[i] >= B || z->vec[L-1-j].coeffs[i] <= -B) {
                return 1;
            }
		}
	}
    return 0;
}

void evaluate_cs1(polyvecl *z, const poly *c, const uint64_t s_table[2*N])
{
    int i, j;
    uint64_t w[N] = {0};
    uint64_t mask_s = 0x8040201008;
    for(i = 0; i < N; i ++)
    {
        if(c->coeffs[i] == 1)
        {
            for(j = 0; j < N; j ++)
            {
                w[j] = w[j] + s_table[j-i+N];
            }
        }
        if(c->coeffs[i] == -1)
        {
            for(j = 0; j < N; j ++)
            {
                w[j] = w[j] + (mask_s - s_table[j-i+N]);
            }
        }
    }
    uint64_t temp;
    for(i = 0; i < N; i ++)
    {
        temp =  w[i];
        for(j = 0; j < L; j ++)
        {
            z->vec[L-1-j].coeffs[i] = ((int32_t)(temp & 0x1FF)-TAU*ETA);//(t mod M) - rU (mod q)
			temp >>= 9;//t = t/M
        }
    }
}



void prepare_s2_table(uint64_t s2_table[2*N],const polyveck *s2)
{
	uint32_t k,j;
    uint64_t temp;
    uint64_t mask_s = 0x1008040201008;


	for(k=0; k<N; k++){

        for(j=0; j<K; j++)
		{
			temp = (uint64_t)(ETA + s2->vec[j].coeffs[k]);
			s2_table[k+N] = (s2_table[k+N]<<9) | (temp);
		}
        s2_table[k] = mask_s - s2_table[k+N];
    }
}
void prepare_s2_table_32(uint32_t s21_table[2*N], uint32_t s22_table[2*N],const polyveck *s2)
{
	uint32_t k,j;
    uint32_t temp;
	uint32_t mask_s2 = 0x201008;
	for(k=0; k<N; k++){

        for(j=0; j<3; j++)
		{
			temp = (uint32_t)(ETA + s2->vec[j].coeffs[k]);
			s21_table[k+N] = (s21_table[k+N]<<9) | (temp);
		}
        for(j=3; j<K; j++)
		{
			temp = (uint32_t)(ETA + s2->vec[j].coeffs[k]);
			s22_table[k+N] = (s22_table[k+N]<<9) | (temp);
		}
        s21_table[k] = mask_s2 - s21_table[k+N];
        s22_table[k] = mask_s2 - s22_table[k+N];
    }
}
void prepare_s2_table_32_avx2(uint32_t s21_table[2*N], uint32_t s22_table[2*N],const polyveck *s2)
{
	uint32_t k,j;
    __m256i s2_coeffs_vec, s21_table_vec, s22_table_vec;
    const __m256i mask_s2 = _mm256_set1_epi32(0x201008);
    const __m256i eta = _mm256_set1_epi32(ETA);
	for(k=0; k<N; k+=8){

        for(j=0; j<3; j++)
		{
            s2_coeffs_vec = _mm256_load_si256((__m256i*)&s2->vec[j].coeffs[k]);
            s2_coeffs_vec = _mm256_add_epi32(eta, s2_coeffs_vec);
            s21_table_vec = _mm256_slli_epi32(s21_table_vec,9);
            s21_table_vec = _mm256_or_si256(s21_table_vec, s2_coeffs_vec);
		}
        for(j=3; j<K; j++)
		{
			s2_coeffs_vec = _mm256_load_si256((__m256i*)&s2->vec[j].coeffs[k]);
            s2_coeffs_vec = _mm256_add_epi32(eta,s2_coeffs_vec);
            s22_table_vec = _mm256_slli_epi32(s22_table_vec,9);
            s22_table_vec = _mm256_or_si256(s22_table_vec, s2_coeffs_vec);
		}
        _mm256_store_si256((__m256i*)&s21_table[k+N],s21_table_vec);
        s21_table_vec = _mm256_sub_epi32(mask_s2, s21_table_vec);
        _mm256_store_si256((__m256i*)&s21_table[k],s21_table_vec);
        _mm256_store_si256((__m256i*)&s22_table[k+N],s22_table_vec);
        s22_table_vec = _mm256_sub_epi32(mask_s2, s22_table_vec);
        _mm256_store_si256((__m256i*)&s22_table[k],s22_table_vec);
    }
}
void evaluate_cs2(
		polyveck *z, 
		poly *c, const uint64_t s2_table[2*N])
{
    uint32_t i,j;
    uint64_t answer0[N]={0};
	uint64_t temp;
    uint64_t mask_s = 0x1008040201008;
    for( i = 0 ; i < N ; i++){
        if(c->coeffs[i] == 1){
            for(j = 0 ; j < N ; j++){
                answer0[j] += s2_table[j-i+N];

            }
        }
        else if(c->coeffs[i] == -1){
            for(j = 0 ; j < N ; j++){
                answer0[j] += mask_s - s2_table[j-i+N];

            }
        }
    }
    
    for(i=0; i<N; i++)
	{
        temp = answer0[i];

        for(j=0; j<K; j++)
		{
			z->vec[K-1-j].coeffs[i] = ((int32_t)(temp & 0x1FF)-TAU*ETA);//(t mod M) - rU (mod q)
			temp >>= 9;//t = t/M

		}
	}

}
int evaluate_cs2_earlycheck_32(
		polyveck *z, polyveck *w0,
		poly *c, uint32_t s21_table[2*N], uint32_t s22_table[2*N], int32_t B)
{
    uint32_t i,j;
    uint32_t answer0[N]={0};
    uint32_t answer1[N]={0};
	uint32_t temp, temp2;
    uint32_t mask_s2 = 0x201008;
    for( i = 0 ; i < N ; i++){
        if(c->coeffs[i] == 1){
            for(j = 0 ; j < N ; j++){
                answer0[j] += s21_table[j-i+N];
                answer1[j] += s22_table[j-i+N];

            }
        }
        else if(c->coeffs[i] == -1){
            for(j = 0 ; j < N ; j++){
                answer0[j] += (mask_s2 - s21_table[j-i+N]);
                answer1[j] += (mask_s2 - s22_table[j-i+N]);

            }
        }
    }
    // int32_t t;
    for(i=0; i<N; i++)
	{
        temp = answer1[i];

        for(j=0; j<3; j++)
		{
			z->vec[K-1-j].coeffs[i] = ((int32_t)(temp & 0x1FF)-TAU*ETA);//(t mod M) - rU (mod q)
			temp >>= 9;//t = t/M
            z->vec[K-1-j].coeffs[i] = w0->vec[K-1-j].coeffs[i] - z->vec[K-1-j].coeffs[i];
            if(z->vec[K-1-j].coeffs[i] >= B || z->vec[K-1-j].coeffs[i] <= -B) {
                return 1;
            }
		}
        temp2 = answer0[i];
        for(j=3; j<K; j++)
		{
			z->vec[K-1-j].coeffs[i] = ((int32_t)(temp2 & 0x1FF)-TAU*ETA);//(t mod M) - rU (mod q)
			temp2 >>= 9;//t = t/M
            z->vec[K-1-j].coeffs[i] = w0->vec[K-1-j].coeffs[i] - z->vec[K-1-j].coeffs[i];
            if(z->vec[K-1-j].coeffs[i] >= B || z->vec[K-1-j].coeffs[i] <= -B) {
                return 1;
            }
		}
	}
    return 0;
}
int evaluate_cs2_earlycheck_32_avx2(
		polyveck *z, polyveck *w0,
		poly *c, uint32_t s21_table[2*N], uint32_t s22_table[2*N], int32_t B)
{
    uint32_t i,j;
    uint32_t answer0[N]={0};
    uint32_t answer1[N]={0};
    for( i = 0 ; i < N ; i++){
        if(c->coeffs[i] == 1){

            add_asm(answer0, &s21_table[N - i]);
            add_asm(answer1, &s22_table[N - i]);
        }
        else if(c->coeffs[i] == -1){

            addmask11_asm(answer0, &s21_table[N - i]);
            addmask11_asm(answer1, &s22_table[N - i]);
        }
    }
    __m256i answer0_vec, answer1_vec, z1_vec, z2_vec, w0_vec, z1abs_vec, z2abs_vec, cmp1_vec,cmp2_vec,cmpeq_vec;
    int cmp1_res, cmp2_res, cmpeq_res;
    const __m256i mask_vec = _mm256_set1_epi32(0x1FF);
    const __m256i taueta_vec = _mm256_set1_epi32(TAU*ETA);
    const __m256i boundB_vec = _mm256_set1_epi32(B);
    for(i=0;i<N; i +=8)
	{
        answer0_vec =_mm256_loadu_si256((__m256i*)&answer0[i]);
        answer1_vec =_mm256_loadu_si256((__m256i*)&answer1[i]);

        for(j=0; j<3; j++)
		{
			z2_vec =_mm256_and_si256(answer1_vec, mask_vec);
            z2_vec =_mm256_sub_epi32(z2_vec, taueta_vec);
            w0_vec = _mm256_loadu_si256((__m256i*)&w0->vec[K-1-j].coeffs[i]);
            z2_vec = _mm256_sub_epi32(w0_vec, z2_vec);
            z2abs_vec = _mm256_abs_epi32(z2_vec);
            cmp2_vec = _mm256_sub_epi32(boundB_vec, z2abs_vec);
            cmpeq_vec = _mm256_cmpeq_epi32(boundB_vec, z2abs_vec);
            cmp2_res = _mm256_movemask_ps((__m256)cmp2_vec);
            cmpeq_res = _mm256_movemask_ps((__m256)cmpeq_vec);
            if(cmp2_res || cmpeq_res)
            {
                return 1;
            }
            answer1_vec = _mm256_srli_epi32(answer1_vec, 9);
            _mm256_storeu_si256((__m256i*)&z->vec[K-1-j].coeffs[i],z2_vec);  
		}
        for(j=3; j<K; j++)
		{
			z1_vec =_mm256_and_si256(answer0_vec, mask_vec);
            z1_vec =_mm256_sub_epi32(z1_vec, taueta_vec);
            w0_vec = _mm256_loadu_si256((__m256i*)&w0->vec[K-1-j].coeffs[i]);
            z1_vec = _mm256_sub_epi32(w0_vec, z1_vec);
            z1abs_vec = _mm256_abs_epi32(z1_vec);
            cmp1_vec = _mm256_sub_epi32(boundB_vec, z1abs_vec);
            cmpeq_vec = _mm256_cmpeq_epi32(boundB_vec, z1abs_vec);
            cmp1_res = _mm256_movemask_ps((__m256)cmp1_vec);
            cmpeq_res = _mm256_movemask_ps((__m256)cmpeq_vec);
            if(cmp1_res || cmpeq_res)
            {
                return 1;
            }
            answer0_vec = _mm256_srli_epi32(answer0_vec, 9);
            _mm256_storeu_si256((__m256i*)&z->vec[K-1-j].coeffs[i],z1_vec);
		}
	}
    return 0;
}
void prepare_t0_table(uint64_t t00_table[4*N], uint64_t t01_table[4*N],const polyveck *t0)
{
	uint32_t k,j;
    uint64_t temp , temp2;
	uint64_t mask_t00, mask_t01;	
	mask_t00 = 0x8000100002000;
	mask_t01 = 0x8000100002000;

	for(k=0; k<N; k++){
        t00_table[k+N] = 0;
        for(j=0; j<3; j++)
		{
			temp = (uint64_t)(0x1000 + t0->vec[j].coeffs[k]);
			temp2 = (uint64_t)(0x1000 - t0->vec[j].coeffs[k]);
			t00_table[k+N] = (t00_table[k+N]<<19) | (temp);
			t00_table[k] = (t00_table[k]<<19) | (temp2);
		}
     
        t00_table[k+3*N] = mask_t00 - t00_table[k+N];
        t00_table[k+2*N] = mask_t00 - t00_table[k];

        t01_table[k+N] = 0;
        for(j=3; j<6; j++)
		{
			temp = (uint64_t)(0x1000 + t0->vec[j].coeffs[k]);
			temp2 = (uint64_t)(0x1000 - t0->vec[j].coeffs[k]);
			t01_table[k+N] = (t01_table[k+N]<<19) | (temp);
			t01_table[k] = (t01_table[k]<<19) | (temp2);
		}
     
        t01_table[k+3*N] = mask_t01 - t01_table[k+N];
        t01_table[k+2*N] = mask_t01 - t01_table[k];

    }

}

void evaluate_ct0(polyveck *z,const poly *c, 
                    const uint64_t t00_table[4*N], const uint64_t t01_table[4*N])
{
    uint32_t i,j;
    uint64_t answer0[N]={0},answer1[N]={0};
	uint64_t temp;

    for( i = 0 ; i < N ; i++){
        if(c->coeffs[i] == 1){
            for(j = 0 ; j < N ; j++){
                answer0[j] += t00_table[j-i+N];
                answer1[j] += t01_table[j-i+N];
            }
        }
        else if(c->coeffs[i] == -1){
            for(j = 0 ; j < N ; j++){
                answer0[j] += t00_table[j-i+3*N];
                answer1[j] += t01_table[j-i+3*N];
            }
        }
    }
    

    for(i=0; i<N; i++)
	{
        temp = answer0[i];
        for(j=0; j<3; j++)
		{
			z->vec[2-j].coeffs[i] = ((int32_t)(temp & 0x7FFFF)-200704);//(t mod M) - rU (mod q)
			temp >>= 19;//t = t/M

		}

        temp = answer1[i];
        for(j=0; j<3; j++)
		{
			z->vec[5-j].coeffs[i] = ((int32_t)(temp & 0x7FFFF)-200704);//(t mod M) - rU (mod q)
			temp >>= 19;//t = t/M
		}
	}
}
int evaluate_cs2_earlycheck(
		polyveck *z, polyveck *w0,
		poly *c, const uint64_t s2_table[2*N], int32_t B)
{
    uint32_t i,j;
    uint64_t answer0[N]={0};
	uint64_t temp;
    uint64_t mask_s = 0x1008040201008;
    for( i = 0 ; i < N ; i++){
        if(c->coeffs[i] == 1){
            for(j = 0 ; j < N ; j++){
                answer0[j] += s2_table[j-i+N];

            }
        }
        else if(c->coeffs[i] == -1){
            for(j = 0 ; j < N ; j++){
                answer0[j] += mask_s - s2_table[j-i+N];

            }
        }
    }
    // int32_t t;
    for(i=0; i<N; i++)
	{
        temp = answer0[i];

        for(j=0; j<K; j++)
		{
			z->vec[K-1-j].coeffs[i] = ((int32_t)(temp & 0x1FF)-TAU*ETA);//(t mod M) - rU (mod q)
			temp >>= 9;//t = t/M
            z->vec[K-1-j].coeffs[i] = w0->vec[K-1-j].coeffs[i] - z->vec[K-1-j].coeffs[i];
            // t = z->vec[K-1-j].coeffs[i] >> 31;
            // t = z->vec[K-1-j].coeffs[i] - (t & 2*z->vec[K-1-j].coeffs[i]);

            // if(t >= B) {
            //     return 1;
            // }
            if(z->vec[K-1-j].coeffs[i] >= B || z->vec[K-1-j].coeffs[i] <= -B) {
                return 1;
            }
		}
	}
    return 0;

}
        

void prepare_t1_table(uint64_t t10_table[4*N], uint64_t t11_table[4*N],const polyveck *t1)
{
	uint32_t k,j;
    uint64_t temp , temp2;
	uint64_t mask_t10, mask_t11;
	mask_t10 = 0x200010000800;
	mask_t11 = 0x200010000800;

	for(k=0; k<N; k++){
        t10_table[k+N] = 0;
        for(j=0; j<3; j++)
		{
			temp = (uint64_t)(0x0400 + t1->vec[j].coeffs[k]);
			temp2 = (uint64_t)(0x0400 - t1->vec[j].coeffs[k]);
			t10_table[k+N] = (t10_table[k+N]<<17) | (temp);
			t10_table[k] = (t10_table[k]<<17) | (temp2);
		}
     
        t10_table[k+3*N] = mask_t10 - t10_table[k+N];
        t10_table[k+2*N] = mask_t10 - t10_table[k];


        t11_table[k+N] = 0;
        for(j=3; j<6; j++)
		{
			temp = (uint64_t)(0x0400 + t1->vec[j].coeffs[k]);
			temp2 = (uint64_t)(0x0400 - t1->vec[j].coeffs[k]);
			t11_table[k+N] = (t11_table[k+N]<<17) | (temp);
			t11_table[k] = (t11_table[k]<<17) | (temp2);
		}
     
        t11_table[k+3*N] = mask_t11 - t11_table[k+N];
        t11_table[k+2*N] = mask_t11 - t11_table[k];
    }

}

void evaluate_ct1(polyveck *z, const poly *c, 
		const uint64_t t10_table[4*N], const uint64_t t11_table[4*N])
{
    uint32_t i,j;
    uint64_t answer0[N]={0},answer1[N]={0};
	uint64_t temp;

    for( i = 0 ; i < N ; i++){
        if(c->coeffs[i] == 1){
            for(j = 0 ; j < N ; j++){
                answer0[j] += t10_table[j-i+N];
                answer1[j] += t11_table[j-i+N];
            }
        }
        else if(c->coeffs[i] == -1){
            for(j = 0 ; j < N ; j++){
                answer0[j] += t10_table[j-i+3*N];
                answer1[j] += t11_table[j-i+3*N];
            }
        }
    }
    
    for(i=0; i<N; i++)
	{
        temp = answer0[i];
        for(j=0; j<3; j++)
		{
			z->vec[2-j].coeffs[i] = ((int32_t)(temp & 0x1FFFF)-50176);//(t mod M) - rU (mod q)
			temp >>= 17;//t = t/M
		}

        temp = answer1[i]; 
        for(j=0; j<3; j++)
		{
			z->vec[5-j].coeffs[i] = ((int32_t)(temp & 0x1FFFF)-50176);//(t mod M) - rU (mod q)
			temp >>= 17;//t = t/M
		}
	}
}
        