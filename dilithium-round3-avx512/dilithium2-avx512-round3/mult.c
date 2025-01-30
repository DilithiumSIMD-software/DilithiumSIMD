#include <stdio.h>
#include <stdint.h>
#include "params.h"
#include "poly.h"
#include "polyvec.h"
#include "mult.h"
#include <immintrin.h>

#define gama 289360691352306692
#define masks 67372036


int evaluate_cs1_cs2_early_check_32_AVX512_opt(polyvecl *z1, polyveck *z2, const poly *c, const uint32_t s1_table[2*N], const uint32_t s2_table[2*N], polyvecl *y, polyveck *w0, int32_t A, int32_t B)
{
    uint32_t i,j;
	__m512i answervec[16];
	__m512i answervec2[16];
    for(i=0;i<16;++i)
    {
        answervec[i]=_mm512_setzero_epi32();
        answervec2[i]=_mm512_setzero_epi32();
    }
    __m512i s1_table_vec,s2_table_vec;
    const __m512i mask=_mm512_set1_epi32(masks);
	//evaluate
    for(i=0;i<N;++i)
    {
        if(c->coeffs[i]==1)
        {
            for(j=0;j<16;++j)
            {
                s1_table_vec=_mm512_loadu_si512(&s1_table[16*j-i+N]);
                answervec[j]=_mm512_add_epi32(answervec[j], s1_table_vec);
                s2_table_vec=_mm512_loadu_si512(&s2_table[16*j-i+N]);
                answervec2[j]=_mm512_add_epi32(answervec2[j], s2_table_vec);
            }
        }
        else if(c->coeffs[i]==-1)
        {
            for(j=0;j<16;++j)
            {
            s1_table_vec=_mm512_loadu_si512(&s1_table[16*j-i+N]);
            s1_table_vec=_mm512_sub_epi32(mask, s1_table_vec);
            answervec[j]=_mm512_add_epi32(answervec[j], s1_table_vec);
            s2_table_vec=_mm512_loadu_si512(&s2_table[16*j-i+N]);
            s2_table_vec=_mm512_sub_epi32(mask, s2_table_vec);
            answervec2[j]=_mm512_add_epi32(answervec2[j], s2_table_vec);            
            }
        }
    }
    //recover
    __m512i answer_vec, answer2_vec, z1_vec, z2_vec, y_vec, w0_vec, z1abs_vec, z2abs_vec;
    unsigned short cmp1_res, cmp2_res;
    const __m512i mask_vec = _mm512_set1_epi32(0xFF);
    const __m512i taueta_vec = _mm512_set1_epi32(TAU*ETA);
    const __m512i boundA_vec = _mm512_set1_epi32(A);
    const __m512i boundB_vec = _mm512_set1_epi32(B);
    for(i=0;i<N; i = i+16)
    {
        answer_vec =answervec[i/16];
        answer2_vec =answervec2[i/16];
        for(j=0; j<4; j++)
		{ 
            z2_vec =_mm512_and_epi32(answer2_vec, mask_vec);
            z2_vec =_mm512_sub_epi32(z2_vec, taueta_vec);
            w0_vec = _mm512_loadu_si512(&w0->vec[K-1-j].coeffs[i]);
            z2_vec = _mm512_sub_epi32(w0_vec, z2_vec);
            z2abs_vec = _mm512_abs_epi32(z2_vec);
            cmp2_res = _mm512_cmp_epi32_mask(z2abs_vec, boundB_vec,_MM_CMPINT_NLT);
            if(cmp2_res)
            {
                return 1;
            }
            z1_vec =_mm512_and_epi32(answer_vec, mask_vec);
            z1_vec =_mm512_sub_epi32(z1_vec, taueta_vec);
            y_vec = _mm512_loadu_si512(&y->vec[L-1-j].coeffs[i]);
            z1_vec = _mm512_add_epi32(z1_vec, y_vec);
            z1abs_vec = _mm512_abs_epi32(z1_vec);
            cmp1_res = _mm512_cmp_epi32_mask(z1abs_vec, boundA_vec,_MM_CMPINT_NLT);
            if(cmp1_res)
            {
                return 1;
            }
            answer_vec = _mm512_srli_epi32(answer_vec, 8);
            answer2_vec = _mm512_srli_epi32(answer2_vec, 8);
            _mm512_storeu_si512(&z1->vec[L-1-j].coeffs[i],z1_vec);
            _mm512_storeu_si512(&z2->vec[K-1-j].coeffs[i],z2_vec);   
		}
    }
	return 0;

}