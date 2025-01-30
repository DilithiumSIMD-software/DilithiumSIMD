#include <stdint.h>
#include "../sign.h"
#include "../poly.h"
#include "../polyvec.h"
#include "../params.h"
#include "cpucycles.h"
#include "speed_print.h"

#define MLEN 59
#define NTESTS 1000
#define masks 67372036
int main(void)
{
  unsigned int i,j;
  __attribute__((aligned(64)))
  uint8_t seedbuf[3*SEEDBYTES];
  __attribute__((aligned(64)))
  __attribute__((aligned(32))) uint32_t s1_table[2*N];
  __attribute__((aligned(32))) uint32_t s2_table[2*N];
  const uint8_t *rho, *rhoprime;
  polyvecl s1;
  polyveck s2;
  polyvecl z,zl;
  poly cp;
  polyveck h,hk;
  randombytes(seedbuf, SEEDBYTES);
  shake256(seedbuf, 3*SEEDBYTES, seedbuf, SEEDBYTES);
  rho = seedbuf;
  rhoprime = seedbuf + SEEDBYTES;
 // key = seedbuf + 2*SEEDBYTES;

  polyvecl_uniform_eta(&s1, rhoprime, 0);
  polyveck_uniform_eta(&s2, rhoprime, L);
  poly_challenge(&cp,rho);

  prepare_s1_table(s1_table, &s1);
  prepare_s2_table(s2_table, &s2);
  uint32_t answer[N]={0};
  uint32_t answer2[N]={0};
    for( i = 0 ; i < N ; i++){
        if(cp.coeffs[i] == 1){

            add_asm(answer, &s1_table[N - i]);
            add_asm(answer2, &s2_table[N - i]);
        }
        else if(cp.coeffs[i] == -1){

            addmask_asm(answer, &s1_table[N - i]);
            addmask_asm(answer2, &s2_table[N - i]);
        }
    }

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
        if(cp.coeffs[i]==1)
        {
            for(j=0;j<16;++j)
            {
                s1_table_vec=_mm512_loadu_si512(&s1_table[16*j-i+N]);
                answervec[j]=_mm512_add_epi32(answervec[j], s1_table_vec);
                s2_table_vec=_mm512_loadu_si512(&s2_table[16*j-i+N]);
                answervec2[j]=_mm512_add_epi32(answervec2[j], s2_table_vec);
            }
        }
        else if(cp.coeffs[i]==-1)
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
        
}


