#include <stdio.h>
#include <stdint.h>
#include "params.h"
#include "poly.h"
#include "mult.h"
#include <immintrin.h>
void prepare_s1_table_32_avx512(uint32_t s11_table[2*N], uint32_t s12_table[2*N],const polyvecl *s1)
{
    uint32_t k,j;
    __m512i s1_coeffs_vec, s11_table_vec, s12_table_vec;
    const __m512i mask_s11 = _mm512_set1_epi32(2101256);
    const __m512i mask_s12 = _mm512_set1_epi32(4104);
    const __m512i eta = _mm512_set1_epi32(ETA);
    for(k=0; k<N; k+=16){
        for(j=0; j<3; j++)
		{
            s1_coeffs_vec = _mm512_load_si512((__m512i*)&s1->vec[j].coeffs[k]);
            s1_coeffs_vec = _mm512_add_epi32(eta, s1_coeffs_vec);
            s11_table_vec = _mm512_slli_epi32(s11_table_vec,9);
            s11_table_vec = _mm512_or_si512(s11_table_vec, s1_coeffs_vec);
		}
        for(j=3; j<L; j++)
		{
			s1_coeffs_vec = _mm512_load_si512((__m512i*)&s1->vec[j].coeffs[k]);
            s1_coeffs_vec = _mm512_add_epi32(eta,s1_coeffs_vec);
            s12_table_vec = _mm512_slli_epi32(s12_table_vec,9);
            s12_table_vec = _mm512_or_si512(s12_table_vec, s1_coeffs_vec);
		}
        _mm512_store_si512((__m512i*)&s11_table[k+N],s11_table_vec);
        s11_table_vec = _mm512_sub_epi32(mask_s11, s11_table_vec);
        _mm512_store_si512((__m512i*)&s11_table[k],s11_table_vec);
        _mm512_store_si512((__m512i*)&s12_table[k+N],s12_table_vec);
        s12_table_vec = _mm512_sub_epi32(mask_s12, s12_table_vec);
        _mm512_store_si512((__m512i*)&s12_table[k],s12_table_vec);
    }
}
int evaluate_cs1_earlycheck_32_avx512_opt(polyvecl *z, polyvecl *y, const poly *c, const uint32_t s11_table[2*N], const uint32_t s12_table[2*N], int32_t B)
{
    int i, j;
	__m512i answervec1[16];
	__m512i answervec2[16];
    for(i=0;i<16;++i)
    {
        answervec1[i]=_mm512_setzero_epi32();
        answervec2[i]=_mm512_setzero_epi32();
    }
    __m512i s11_table_vec,s12_table_vec;
    const __m512i mask1=_mm512_set1_epi32(0x201008);
    const __m512i mask2=_mm512_set1_epi32(0x1008);
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


     __m512i w_vec, w2_vec, z1_vec, z2_vec, y_vec, zabs_vec;
     __mmask16 cmp1_res, cmp2_res;
    const __m512i mask_vec = _mm512_set1_epi32(0x1FF);
    const __m512i taueta_vec = _mm512_set1_epi32(TAU*ETA);
    const __m512i boundB_vec = _mm512_set1_epi32(B);
    for(i=0; i<N; i+=16)
	{
        w_vec =answervec1[i/16];
        w2_vec =answervec2[i/16];

        for(j=0; j<2 ; j++)
		{
			z2_vec =_mm512_and_si512(w2_vec, mask_vec);
            z2_vec =_mm512_sub_epi32(z2_vec, taueta_vec);
            y_vec = _mm512_loadu_si512((__m512i*)&y->vec[L-1-j].coeffs[i]);
            z2_vec = _mm512_add_epi32(z2_vec, y_vec);
            zabs_vec = _mm512_abs_epi32(z2_vec);
            cmp1_res = _mm512_cmp_epi32_mask(zabs_vec, boundB_vec,5);
            if((int32_t)cmp1_res)
            {
                return 1;
            }
            w2_vec = _mm512_srli_epi32(w2_vec, 9);
            _mm512_storeu_si512((__m512i*)&z->vec[L-1-j].coeffs[i],z2_vec);
		}
        for(j=2;j < L; j++)
        {
            z1_vec =_mm512_and_si512(w_vec, mask_vec);
            z1_vec =_mm512_sub_epi32(z1_vec, taueta_vec);
            y_vec = _mm512_loadu_si512((__m512i*)&y->vec[L-1-j].coeffs[i]);
            z1_vec = _mm512_add_epi32(z1_vec, y_vec);
            zabs_vec = _mm512_abs_epi32(z1_vec);
            cmp2_res = _mm512_cmp_epi32_mask(zabs_vec, boundB_vec,5);
            if((int32_t)cmp2_res)
            {
                return 1;
            }
            w_vec = _mm512_srli_epi32(w_vec, 9);
            _mm512_storeu_si512((__m512i*)&z->vec[L-1-j].coeffs[i],z1_vec); 
        }
	}
    return 0;
}
void prepare_s2_table_32_avx512(uint32_t s21_table[2*N], uint32_t s22_table[2*N],const polyveck *s2)
{
	uint32_t k,j;
    __m512i s2_coeffs_vec, s21_table_vec, s22_table_vec;
    const __m512i mask_s2 = _mm512_set1_epi32(0x201008);
    const __m512i eta = _mm512_set1_epi32(ETA);
	for(k=0; k<N; k+=16){

        for(j=0; j<3; j++)
		{
            s2_coeffs_vec = _mm512_load_si512((__m512i*)&s2->vec[j].coeffs[k]);
            s2_coeffs_vec = _mm512_add_epi32(eta, s2_coeffs_vec);
            s21_table_vec = _mm512_slli_epi32(s21_table_vec,9);
            s21_table_vec = _mm512_or_si512(s21_table_vec, s2_coeffs_vec);
		}
        for(j=3; j<K; j++)
		{
			s2_coeffs_vec = _mm512_load_si512((__m512i*)&s2->vec[j].coeffs[k]);
            s2_coeffs_vec = _mm512_add_epi32(eta,s2_coeffs_vec);
            s22_table_vec = _mm512_slli_epi32(s22_table_vec,9);
            s22_table_vec = _mm512_or_si512(s22_table_vec, s2_coeffs_vec);
		}
        _mm512_store_si512((__m512i*)&s21_table[k+N],s21_table_vec);
        s21_table_vec = _mm512_sub_epi32(mask_s2, s21_table_vec);
        _mm512_store_si512((__m512i*)&s21_table[k],s21_table_vec);
        _mm512_store_si512((__m512i*)&s22_table[k+N],s22_table_vec);
        s22_table_vec = _mm512_sub_epi32(mask_s2, s22_table_vec);
        _mm512_store_si512((__m512i*)&s22_table[k],s22_table_vec);
    }
}
int evaluate_cs2_earlycheck_32_avx512_opt(polyveck *z, polyveck *w0,const poly *c, const uint32_t s21_table[2*N], const uint32_t s22_table[2*N], int32_t B)
{
    uint32_t i,j;
	__m512i answervec1[16];
	__m512i answervec2[16];
    for(i=0;i<16;++i)
    {
        answervec1[i]=_mm512_setzero_epi32();
        answervec2[i]=_mm512_setzero_epi32();
    }
    __m512i s21_table_vec,s22_table_vec;
    const __m512i mask=_mm512_set1_epi32(0x201008);
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
     __m512i answer0_vec, answer1_vec, z1_vec, z2_vec, w0_vec, z1abs_vec, z2abs_vec;
     __mmask16 cmp1_res, cmp2_res;
    const __m512i mask_vec = _mm512_set1_epi32(0x1FF);
    const __m512i taueta_vec = _mm512_set1_epi32(TAU*ETA);
    const __m512i boundB_vec = _mm512_set1_epi32(B);
    for(i=0;i<N; i +=16)
	{
        answer0_vec =answervec1[i/16];
        answer1_vec =answervec2[i/16];

        for(j=0; j<3; j++)
		{
			z2_vec =_mm512_and_si512(answer1_vec, mask_vec);
            z2_vec =_mm512_sub_epi32(z2_vec, taueta_vec);
            w0_vec = _mm512_loadu_si512((__m512i*)&w0->vec[K-1-j].coeffs[i]);
            z2_vec = _mm512_sub_epi32(w0_vec, z2_vec);
            z2abs_vec = _mm512_abs_epi32(z2_vec);
            cmp2_res = _mm512_cmp_epi32_mask(z2abs_vec, boundB_vec,5);
            if((int32_t)cmp2_res)
            {
                return 1;
            }
            answer1_vec = _mm512_srli_epi32(answer1_vec, 9);
            _mm512_storeu_si512((__m512i*)&z->vec[K-1-j].coeffs[i],z2_vec);  
		}
        for(j=3; j<K; j++)
		{
			z1_vec =_mm512_and_si512(answer0_vec, mask_vec);
            z1_vec =_mm512_sub_epi32(z1_vec, taueta_vec);
            w0_vec = _mm512_loadu_si512((__m512i*)&w0->vec[K-1-j].coeffs[i]);
            z1_vec = _mm512_sub_epi32(w0_vec, z1_vec);
            z1abs_vec = _mm512_abs_epi32(z1_vec);
            cmp1_res = _mm512_cmp_epi32_mask(z1abs_vec, boundB_vec,5);
            if((int32_t)cmp1_res)
            {
                return 1;
            }
            answer0_vec = _mm512_srli_epi32(answer0_vec, 9);
            _mm512_storeu_si512((__m512i*)&z->vec[K-1-j].coeffs[i],z1_vec);
		}
	}
    return 0;
}