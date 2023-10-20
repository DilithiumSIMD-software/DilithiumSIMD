#include <stdio.h>
#include <stdint.h>
#include "params.h"
#include "mult.h"
#include "poly.h"
#include "polyvec.h"
#include <immintrin.h>
#define gama1 1130315200594948
#define gama2 289360691352306692
int evaluate_cs2_earlycheck_AVX512_opt(polyveck *z2,  const poly *c, const uint32_t s21_table[2*N], const uint32_t s22_table[2*N],  polyveck *w0, int32_t B)
{
    uint32_t i,j;
    __m512i answervec1[16];
	__m512i answervec2[16];

    //evaluate
    for(i=0;i<16;++i)
    {
        answervec1[i]=_mm512_setzero_epi32();
        answervec2[i]=_mm512_setzero_epi32();
    }
    __m512i s21_table_vec,s22_table_vec;
    const __m512i mask=_mm512_set1_epi32(0X04040404);
    for(i = 0; i < N; i ++)
    {
        if(c->coeffs[i] == 1)
        {
            for(j=0;j<16;++j)
            {
                s21_table_vec=_mm512_loadu_si512(&s21_table[16*j-i+N]);
                answervec1[j]=_mm512_add_epi32(answervec1[j], s21_table_vec);
                s22_table_vec=_mm512_loadu_si512(&s22_table[16*j-i+N]);
                answervec2[j]=_mm512_add_epi32(answervec2[j], s22_table_vec);            
            }
        }
        else if(c->coeffs[i] == -1)
        {
            for(j=0;j<16;++j)
            {
                s21_table_vec=_mm512_loadu_si512(&s21_table[16*j-i+N]);
                s21_table_vec=_mm512_sub_epi32(mask, s21_table_vec);
                answervec1[j]=_mm512_add_epi32(answervec1[j], s21_table_vec);
                s22_table_vec=_mm512_loadu_si512(&s22_table[16*j-i+N]);
                s22_table_vec=_mm512_sub_epi32(mask, s22_table_vec);
                answervec2[j]=_mm512_add_epi32(answervec2[j], s22_table_vec);            
            }
        }
    }

    //recover
    __m512i answer_vec, answer2_vec, z2_vec,  w0_vec,  z2abs_vec;
    unsigned short cmp_res;
    const __m512i mask_vec = _mm512_set1_epi32(0xFF);
    const __m512i taueta_vec = _mm512_set1_epi32(TAU*ETA);
    const __m512i boundB_vec = _mm512_set1_epi32(B);
    for(i=0;i<N; i = i+16)
    {
        answer_vec =answervec1[i/16];
        answer2_vec =answervec2[i/16];
        for(j=0; j<4; ++j)
		{ 
            z2_vec =_mm512_and_epi32(answer2_vec, mask_vec);
            z2_vec =_mm512_sub_epi32(z2_vec, taueta_vec);
            w0_vec = _mm512_loadu_si512(&w0->vec[K-1-j].coeffs[i]);
            z2_vec = _mm512_sub_epi32(w0_vec, z2_vec);
            z2abs_vec = _mm512_abs_epi32(z2_vec);
            cmp_res = _mm512_cmp_epi32_mask(z2abs_vec, boundB_vec,_MM_CMPINT_NLT);
            if(cmp_res)
            {
                return 1;
            }
            answer2_vec = _mm512_srli_epi32(answer2_vec, 8);
            _mm512_storeu_si512(&z2->vec[K-1-j].coeffs[i],z2_vec);   
        }
        for(j=4;j<8;++j)
        {
            z2_vec =_mm512_and_epi32(answer_vec, mask_vec);
            z2_vec =_mm512_sub_epi32(z2_vec, taueta_vec);
            w0_vec = _mm512_loadu_si512(&w0->vec[K-1-j].coeffs[i]);
            z2_vec = _mm512_sub_epi32(w0_vec, z2_vec);
            z2abs_vec = _mm512_abs_epi32(z2_vec);
            cmp_res = _mm512_cmp_epi32_mask(z2abs_vec, boundB_vec,_MM_CMPINT_NLT);
            if(cmp_res)
            {
                return 1;
            }
            answer_vec = _mm512_srli_epi32(answer_vec, 8);
            _mm512_storeu_si512(&z2->vec[K-1-j].coeffs[i],z2_vec);  

        }
    }
	return 0;



}
int evaluate_cs1_earlycheck_AVX512_opt(polyvecl *z,  const poly *c, const uint32_t s11_table[2*N], const uint32_t s12_table[2*N],  polyvecl *y,  int32_t A)
{
    uint32_t i,j;
	__m512i answervec1[16];
	__m512i answervec2[16];
	
	//evaluate
    for(i=0;i<16;++i)
    {
        answervec1[i]=_mm512_setzero_epi32();
        answervec2[i]=_mm512_setzero_epi32();
    }
    __m512i s11_table_vec,s12_table_vec;
    const __m512i mask1=_mm512_set1_epi32(0X04040404);
    const __m512i mask2=_mm512_set1_epi32(0X040404);
    for(i = 0; i < N; i ++)
    {
        if(c->coeffs[i] == 1)
        {
            for(j=0;j<16;++j)
            {
                s11_table_vec=_mm512_loadu_si512(&s11_table[16*j-i+N]);
                answervec1[j]=_mm512_add_epi32(answervec1[j], s11_table_vec);
                s12_table_vec=_mm512_loadu_si512(&s12_table[16*j-i+N]);
                answervec2[j]=_mm512_add_epi32(answervec2[j], s12_table_vec);            
            }
        }
        else if(c->coeffs[i] == -1)
        {
            for(j=0;j<16;++j)
            {
                s11_table_vec=_mm512_loadu_si512(&s11_table[16*j-i+N]);
                s11_table_vec=_mm512_sub_epi32(mask1, s11_table_vec);
                answervec1[j]=_mm512_add_epi32(answervec1[j], s11_table_vec);
                s12_table_vec=_mm512_loadu_si512(&s12_table[16*j-i+N]);
                s12_table_vec=_mm512_sub_epi32(mask2, s12_table_vec);
                answervec2[j]=_mm512_add_epi32(answervec2[j], s12_table_vec);            
            }
        }
    }
    //recover
    __m512i answer_vec, answer2_vec, z1_vec, y_vec,  z1abs_vec;
    unsigned short cmp_res;
    const __m512i mask_vec = _mm512_set1_epi32(0xFF);
    const __m512i taueta_vec = _mm512_set1_epi32(TAU*ETA);
    const __m512i boundA_vec = _mm512_set1_epi32(A);
    for(i=0;i<N; i = i+16)
    {
        answer_vec =answervec1[i/16];
        answer2_vec =answervec2[i/16];

        for(j=0; j<3; j++)
		{ 
            z1_vec =_mm512_and_epi32(answer2_vec, mask_vec);
            z1_vec =_mm512_sub_epi32(z1_vec, taueta_vec);
            y_vec = _mm512_loadu_si512(&y->vec[L-1-j].coeffs[i]);
            z1_vec = _mm512_add_epi32( z1_vec,y_vec);
            z1abs_vec = _mm512_abs_epi32(z1_vec);
            cmp_res = _mm512_cmp_epi32_mask(z1abs_vec, boundA_vec,_MM_CMPINT_NLT);
            if(cmp_res)
            {
                return 1;
            }
            answer2_vec = _mm512_srli_epi32(answer2_vec, 8);
            _mm512_storeu_si512(&z->vec[L-1-j].coeffs[i],z1_vec); 
		}
        for(j=3; j<7; j++)
		{ 
            z1_vec =_mm512_and_epi32(answer_vec, mask_vec);
            z1_vec =_mm512_sub_epi32(z1_vec, taueta_vec);
            y_vec = _mm512_loadu_si512(&y->vec[L-1-j].coeffs[i]);
            z1_vec = _mm512_add_epi32(z1_vec,y_vec);
            z1abs_vec = _mm512_abs_epi32(z1_vec);
            cmp_res = _mm512_cmp_epi32_mask(z1abs_vec, boundA_vec,_MM_CMPINT_NLT);
            if(cmp_res)
            {
                return 1;
            }
            answer_vec = _mm512_srli_epi32(answer_vec, 8);
            _mm512_storeu_si512(&z->vec[L-1-j].coeffs[i],z1_vec); 
		}
    }
	return 0;

}