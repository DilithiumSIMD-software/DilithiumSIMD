#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>
#include "params.h"
#include "mult.h"
#include "poly.h"
#include "polyvec.h"
#define gama1 1130315200594948
#define gama2 289360691352306692
int evaluate_cs2_earlycheck_avx2(polyveck *z2,  const poly *c, const uint32_t s21_table[2*N], const uint32_t s22_table[2*N],  polyveck *w0, int32_t B)
{
    uint32_t i,j;
    uint32_t answer[N]={0};
    uint32_t answer2[N]={0};
	
	//evaluate
    for( i = 0 ; i < N ; i++){
        if(c->coeffs[i] == 1){

            add_asm(answer, &s21_table[N - i]);
            add_asm(answer2, &s22_table[N - i]);
        }
        else if(c->coeffs[i] == -1){

            addmask_asm(answer, &s21_table[N - i]);
            addmask_asm(answer2, &s22_table[N - i]);
        }
    }
    __m256i answer_vec, answer2_vec,z2_vec,w0_vec, z2abs_vec, cmp2_vec,cmpeq_vec;
    int  cmp2_res, cmpeq_res;
    const __m256i mask_vec = _mm256_set1_epi32(0xFF);
    const __m256i taueta_vec = _mm256_set1_epi32(TAU*ETA);
    const __m256i boundB_vec = _mm256_set1_epi32(B);
    for(i=0;i<N; i +=8)
    {
        answer_vec =_mm256_loadu_si256((__m256i*)&answer[i]);
        answer2_vec =_mm256_loadu_si256((__m256i*)&answer2[i]);
        for(j=0; j<4; j++)
		{ 
            z2_vec =_mm256_and_si256(answer2_vec, mask_vec);
            z2_vec =_mm256_sub_epi32(z2_vec, taueta_vec);
            w0_vec = _mm256_loadu_si256((__m256i*)&w0->vec[K-1-j].coeffs[i]);
            z2_vec = _mm256_sub_epi32(w0_vec,z2_vec);
            z2abs_vec = _mm256_abs_epi32(z2_vec);
            cmp2_vec = _mm256_sub_epi32(boundB_vec, z2abs_vec);
            cmp2_res = _mm256_movemask_ps((__m256)cmp2_vec);
            cmpeq_vec = _mm256_cmpeq_epi32(boundB_vec, z2abs_vec);
            cmpeq_res = _mm256_movemask_ps((__m256)cmpeq_vec);
            if(cmp2_res || cmpeq_res)
            {
                return 1;
            }
            answer2_vec = _mm256_srli_epi32(answer2_vec, 8);
            _mm256_storeu_si256((__m256i*)&z2->vec[K-1-j].coeffs[i],z2_vec); 
		}
        for(j = 4;j<8;j++)
        {
            z2_vec =_mm256_and_si256(answer_vec, mask_vec);
            z2_vec =_mm256_sub_epi32(z2_vec, taueta_vec);
            w0_vec = _mm256_loadu_si256((__m256i*)&w0->vec[K-1-j].coeffs[i]);
            z2_vec = _mm256_sub_epi32(w0_vec,z2_vec);
            z2abs_vec = _mm256_abs_epi32(z2_vec);
            cmp2_vec = _mm256_sub_epi32(boundB_vec, z2abs_vec);
            cmp2_res = _mm256_movemask_ps((__m256)cmp2_vec);
            cmpeq_vec = _mm256_cmpeq_epi32(boundB_vec, z2abs_vec);
            cmpeq_res = _mm256_movemask_ps((__m256)cmpeq_vec);
            if(cmp2_res || cmpeq_res)
            {
                return 1;
            }
            answer_vec = _mm256_srli_epi32(answer_vec, 8);
            _mm256_storeu_si256((__m256i*)&z2->vec[K-1-j].coeffs[i],z2_vec);
        }
    }
	return 0;
}
int evaluate_cs1_earlycheck_avx2(polyvecl *z,  const poly *c, const uint32_t s11_table[2*N], const uint32_t s12_table[2*N],  polyvecl *y,  int32_t A)
{
    uint32_t i,j;
    uint32_t answer[N]={0};
    uint32_t answer2[N]={0};
	
	//evaluate
    for( i = 0 ; i < N ; i++){
        if(c->coeffs[i] == 1){

            add_asm(answer, &s11_table[N - i]);
            add_asm(answer2, &s12_table[N - i]);
        }
        else if(c->coeffs[i] == -1){

            addmask_asm(answer, &s11_table[N - i]);
            addmask_asm2(answer2, &s12_table[N - i]);
        }
    }
    __m256i answer_vec, answer2_vec, z1_vec, y_vec,  z1abs_vec, cmp1_vec,cmpeq_vec;
    int cmp1_res,  cmpeq_res;
    const __m256i mask_vec = _mm256_set1_epi32(0xFF);
    const __m256i taueta_vec = _mm256_set1_epi32(TAU*ETA);
    const __m256i boundA_vec = _mm256_set1_epi32(A);
    for(i=0;i<N; i +=8)
    {
        answer_vec =_mm256_loadu_si256((__m256i*)&answer[i]);
        answer2_vec =_mm256_loadu_si256((__m256i*)&answer2[i]);
        for(j=0; j<3; j++)
		{ 
            z1_vec =_mm256_and_si256(answer2_vec, mask_vec);
            z1_vec =_mm256_sub_epi32(z1_vec, taueta_vec);
            y_vec = _mm256_loadu_si256((__m256i*)&y->vec[L-1-j].coeffs[i]);
            z1_vec = _mm256_add_epi32(z1_vec, y_vec);
            z1abs_vec = _mm256_abs_epi32(z1_vec);
            cmp1_vec = _mm256_sub_epi32(boundA_vec, z1abs_vec);
            cmp1_res = _mm256_movemask_ps((__m256)cmp1_vec);
            cmpeq_vec = _mm256_cmpeq_epi32(boundA_vec, z1abs_vec);
            cmpeq_res = _mm256_movemask_ps((__m256)cmpeq_vec);
            if(cmp1_res || cmpeq_res)
            {
                return 1;
            }
            answer2_vec = _mm256_srli_epi32(answer2_vec, 8);
            _mm256_storeu_si256((__m256i*)&z->vec[L-1-j].coeffs[i],z1_vec);
		}
        for(j = 3;j<7;j++)
        {
            z1_vec =_mm256_and_si256(answer_vec, mask_vec);
            z1_vec =_mm256_sub_epi32(z1_vec, taueta_vec);
            y_vec = _mm256_loadu_si256((__m256i*)&y->vec[L-1-j].coeffs[i]);
            z1_vec = _mm256_add_epi32(z1_vec, y_vec);
            z1abs_vec = _mm256_abs_epi32(z1_vec);
            cmp1_vec = _mm256_sub_epi32(boundA_vec, z1abs_vec);
            cmp1_res = _mm256_movemask_ps((__m256)cmp1_vec);
            cmpeq_vec = _mm256_cmpeq_epi32(boundA_vec, z1abs_vec);
            cmpeq_res = _mm256_movemask_ps((__m256)cmpeq_vec);
            if(cmp1_res || cmpeq_res)
            {
                return 1;
            }
            answer_vec = _mm256_srli_epi32(answer_vec, 8);
            _mm256_storeu_si256((__m256i*)&z->vec[L-1-j].coeffs[i],z1_vec);
        }
    }
	return 0;
}
void prepare_t0_table(uint64_t t00_table[4*N], uint64_t t01_table[4*N],uint64_t t02_table[4*N], polyveck *t0)
{
	uint32_t k,j;
    uint64_t temp , temp2;
	uint64_t mask_t00, mask_t01, mask_t02;	
	mask_t00 = 0x8000100002000;
	mask_t01 = 0x8000100002000;
	mask_t02 = 0x100002000; 

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

        t02_table[k+N] = 0;
        for(j=6; j<8; j++)
		{
			temp = (uint64_t)(0x1000 + t0->vec[j].coeffs[k]);
			temp2 = (uint64_t)(0x1000 - t0->vec[j].coeffs[k]);
			t02_table[k+N] = (t02_table[k+N]<<19) | (temp);
			t02_table[k] = (t02_table[k]<<19) | (temp2);
		}
     
        t02_table[k+3*N] = mask_t02 - t02_table[k+N];
        t02_table[k+2*N] = mask_t02 - t02_table[k];
    }

}


void evaluate_ct0(
		polyveck *z,const poly *c,
		const uint64_t t00_table[4*N], const uint64_t t01_table[4*N], const uint64_t t02_table[4*N])
{
    uint32_t i,j;
    uint64_t answer0[N]={0},answer1[N]={0},answer2[N]={0};
	uint64_t temp;

    for( i = 0 ; i < N ; i++){
        if(c->coeffs[i] == 1){
            for(j = 0 ; j < N ; j++){
                answer0[j] += t00_table[j-i+N];
                answer1[j] += t01_table[j-i+N];
                answer2[j] += t02_table[j-i+N];
            }
        }
        else if(c->coeffs[i] == -1){
            for(j = 0 ; j < N ; j++){
                answer0[j] += t00_table[j-i+3*N];
                answer1[j] += t01_table[j-i+3*N];
                answer2[j] += t02_table[j-i+3*N];
            }
        }
    }


    for(i=0; i<N; i++)
	{
        temp = answer0[i];
        for(j=0; j<3; j++)
		{
			z->vec[2-j].coeffs[i] = ((int32_t)(temp & 0x7FFFF)-245760);//(t mod M) - rU (mod q)
			temp >>= 19;//t = t/M
		}

        temp = answer1[i];
        for(j=0; j<3; j++)
		{
			z->vec[5-j].coeffs[i] = ((int32_t)(temp & 0x7FFFF)-245760);//(t mod M) - rU (mod q)
			temp >>= 19;//t = t/M
		}

        temp = answer2[i];
        for(j=0; j<2; j++)
		{
			z->vec[7-j].coeffs[i] = ((int32_t)(temp & 0x7FFFF)-245760);//(t mod M) - rU (mod q)
			temp >>= 19;//t = t/M
		}
	}
}
    