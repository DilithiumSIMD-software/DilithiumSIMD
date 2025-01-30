//#include "immintrin.h"
//#include "emmintrin.h"
//#include <string.h>
//#include <stdio.h>
//#include <stdint.h>
#include "SHA-3_256.h"

/*#ifdef __INTEL_COMPILER
#define ALIGN __declspec(align(32))
#else
#define ALIGN __attribute__ ((aligned (64)))
#endif
*/

#define load(M,rsiz)	\
	x[0] = _mm512_xor_si512(x[0],_mm512_maskz_loadu_epi64(0x1F,(__m512i*)M)+0);\
	M=M + 40;\
	if(rsiz == 72){\
		x[1] = _mm512_xor_si512(x[1],_mm512_maskz_loadu_epi64(0xF,(__m512i*)M)+0);	\
		M+=32;\
	}else if(rsiz >=104){\
		x[1] = _mm512_xor_si512(x[1],_mm512_maskz_loadu_epi64(0x1F,(__m512i*)M)+0);	\
		M+=40;\
	}\
	if(rsiz == 104){\
		x[2] = _mm512_xor_si512(x[2],_mm512_maskz_loadu_epi64(0x07,(__m512i*)M)+0);	\
		M+=24;\
	}else if (rsiz >= 136){\
		x[2] = _mm512_xor_si512(x[2],_mm512_maskz_loadu_epi64(0x1F,(__m512i*)M)+0);	\
		M+=40;\
	}\
	if(rsiz == 136){\
		x[3] = _mm512_xor_si512(x[3],_mm512_maskz_loadu_epi64(0x03,(__m512i*)M)+0);	\
		M+=16;\
	}else if (rsiz == 144){\
		x[3] = _mm512_xor_si512(x[3],_mm512_maskz_loadu_epi64(0x07,(__m512i*)M)+0);	\
		M+=24;\
	}else if(rsiz == 168){\
		x[3] = _mm512_xor_si512(x[3],_mm512_maskz_loadu_epi64(0x1F,(__m512i*)M)+0);	\
		M+=40;\
		x[4] = _mm512_xor_si512(x[4],_mm512_maskz_loadu_epi64(0x01,(__m512i*)M)+0);	\
		M+=8;\
	}\


void keccakF(__m512i* x, int rnd){
	const uint64_t RC[24] = {
	    0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
	    0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
	    0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
	    0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
	    0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
	    0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
	    0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
	    0x8000000000008080, 0x0000000080000001, 0x8000000080008008
	};
	const __m512i r0 = _mm512_set_epi64(0, 0, 0, 27, 28, 62, 1, 0);
	const __m512i r1 = _mm512_set_epi64(0, 0, 0, 20, 55, 6, 44, 36);
	const __m512i r2 = _mm512_set_epi64(0, 0, 0, 39, 25, 43, 10, 3);
	const __m512i r3 = _mm512_set_epi64(0, 0, 0, 8, 21, 15, 45, 41);
	const __m512i r4 = _mm512_set_epi64(0, 0, 0, 14, 56, 61, 2, 18);

	const __m512i t0 = _mm512_set_epi64(0, 0, 0, 2, 4, 1, 3, 0);
	const __m512i t1 = _mm512_set_epi64(0, 0, 0, 3, 0, 2, 4, 1);
	const __m512i t2 = _mm512_set_epi64(0, 0, 0, 4, 1, 3, 0, 2);
	const __m512i t3 = _mm512_set_epi64(0, 0, 0, 0, 2, 4, 1, 3);
	const __m512i t4 = _mm512_set_epi64(0, 0, 0, 1, 3, 0, 2, 4);
	
	const __m512i p1 = _mm512_set_epi64(3, 4, 10, 2, 9, 1, 8, 0);
	const __m512i p2 = _mm512_set_epi64(10, 2, 4, 3, 8, 0, 9, 1);
	const __m512i p3 = _mm512_set_epi64(0, 0, 0, 0, 11, 3, 12, 4);
	const __m512i p4 = _mm512_set_epi64(12, 11, 0, 8, 3, 2, 1, 0);
	const __m512i p5 = _mm512_set_epi64(0, 0, 0, 9, 1, 0, 3, 2);
	const __m512i p6 = _mm512_set_epi64(0, 0, 0, 10, 7, 6, 5, 4);
	const __m512i p7 = _mm512_set_epi64(0, 0, 0, 11, 3, 4, 2, 7);
	const __m512i p8 = _mm512_set_epi64(0, 0, 0, 4, 9, 5, 8, 6);

	const __m512i p11 = _mm512_set_epi64(0, 0, 0, 3, 2, 1, 0, 4);
	const __m512i p12 = _mm512_set_epi64(0, 0, 0, 0, 4, 3, 2, 1);


	
	uint64_t* rc = (uint64_t*)RC;
	int i;
	__m512i c1,c2,c3,c4,c5;

	for(i=0;i<rnd;i++){

	/*theta step*/
	c1 = _mm512_ternarylogic_epi64(x[0],x[1],x[2],0x96);
	c1 = _mm512_ternarylogic_epi64(c1,x[3],x[4],0x96);

	c2 = _mm512_permutexvar_epi64(p11, c1);
        c1 = _mm512_permutexvar_epi64(p12, c1);
        c1 = _mm512_rol_epi64(c1,0x01);

	x[0] = _mm512_ternarylogic_epi64(c1,c2,x[0],0x96);
	x[1] = _mm512_ternarylogic_epi64(c1,c2,x[1],0x96);
	x[2] = _mm512_ternarylogic_epi64(c1,c2,x[2],0x96);
	x[3] = _mm512_ternarylogic_epi64(c1,c2,x[3],0x96);
	x[4] = _mm512_ternarylogic_epi64(c1,c2,x[4],0x96);
	/*theta step*/


	/*rho and pi step*/
	x[0] = _mm512_rolv_epi64(x[0],r0);/*15-5-20-10-0*/
	x[1] = _mm512_rolv_epi64(x[1],r1);/*6-21-11-1-16*/
	x[2] = _mm512_rolv_epi64(x[2],r2);/*22-12-2-17-7*/
	x[3] = _mm512_rolv_epi64(x[3],r3);/*13-3-18-8-23*/
	x[4] = _mm512_rolv_epi64(x[4],r4);/*4-19-9-24-14*/

	x[0] = _mm512_permutexvar_epi64(t0, x[0]);
        x[1] = _mm512_permutexvar_epi64(t1, x[1]);
        x[2] = _mm512_permutexvar_epi64(t2, x[2]);
        x[3] = _mm512_permutexvar_epi64(t3, x[3]);
        x[4] = _mm512_permutexvar_epi64(t4, x[4]);
	
	/*rho and pi step*/

	/*chi step*/
	c1 = x[0];
	c2 = x[1];
	x[0] = _mm512_ternarylogic_epi64(x[0],x[1],x[2],0xD2);
	x[1] = _mm512_ternarylogic_epi64(x[1],x[2],x[3],0xD2);
	x[2] = _mm512_ternarylogic_epi64(x[2],x[3],x[4],0xD2);
	x[3] = _mm512_ternarylogic_epi64(x[3],x[4],c1,0xD2);
	x[4] = _mm512_ternarylogic_epi64(x[4],c1,c2,0xD2);

	c1  = _mm512_permutex2var_epi64(x[0],p1,x[1]);/*15-20-11-10-6-5-1-0*/
	c2  = _mm512_permutex2var_epi64(x[2],p2,x[3]);/*13-12-22-17-3-2-8-7*/

	c3  = _mm512_mask_blend_epi64(0xCC,c1,c2);/*13-12-11-10-3-2-1-0*/
	c4  = _mm512_mask_blend_epi64(0x33,c1,c2);/*15-20-22-17-6-5-8-7*/

	c5  = _mm512_permutex2var_epi64(x[1],p3,x[3]);/*x-x-x-x-18-16-23-21*/
	x[0]= _mm512_permutex2var_epi64(c3,p4,x[4]);/*x-x-x-4-3-2-1-0*/
	
	x[1]= _mm512_permutex2var_epi64(c4,p5,x[4]);/*x-x-x-9-8-7-6-5*/
	x[2]= _mm512_permutex2var_epi64(c3,p6,x[4]);/*x-x-x-14-13-12-11-10*/

	c2  = _mm512_mask_blend_epi64(0x10,c4,x[4]);/*x-20-22-24-x-x-x-x*/
	c3  = _mm512_mask_blend_epi64(0x0F,c4,c5);/*15-x-x-17-18-16-x-x*/

	x[3]= _mm512_permutex2var_epi64(c3,p7,x[4]);/*x-x-x-19-18-17-16-15*/
	x[4]= _mm512_permutex2var_epi64(c2,p8,c5);/*x-x-x-24-23-22-21-20*/


	/*chi step*/
	

	/*iota step*/
	c1 = _mm512_maskz_loadu_epi64(0x01,(__m512i*)rc);
	rc+=1;
	x[0] = _mm512_xor_si512(x[0],c1);

	/*iota step*/


	}
}

int keccak_avx512(const uint8_t *in, int inlen, uint8_t *md, int r){
	
    uint8_t * in_temp = (uint8_t*)in;
    uint8_t *t;
    __m512i x[5];

    ALIGN uint8_t temp[144];
    memset(temp, 0, 144*sizeof(uint8_t));
    x[0] = _mm512_setzero_si512();
    x[1] = _mm512_setzero_si512();
    x[2] = _mm512_setzero_si512();
    x[3] = _mm512_setzero_si512();
    x[4] = _mm512_setzero_si512();
   int rsiz = 200 - 2*r;
   for ( ; inlen >= rsiz; inlen -= rsiz) {
         load(in_temp, rsiz);
/*	 print_512(x[0]);
	 print_512(x[1]);
	 print_512(x[2]);
	 print_512(x[3]);
	 print_512(x[4]);
	 getchar();*/
    	 keccakF(x, 24);
   }

   // last block and padding
    memcpy(temp, in_temp, inlen);
    temp[inlen++] = 0x06;           // XXX Padding Changed from Keccak 3.0
    memset(temp + inlen, 0, rsiz - inlen);
    temp[rsiz - 1] |= 0x80;

    t = temp;
    load(t,144);
    keccakF(x, 24);
   if(rsiz == 144 || rsiz == 168) 
	 _mm512_mask_storeu_epi32((__m512*)md,0x7F,x[0]);
   else if(rsiz == 136)
   	_mm512_mask_storeu_epi64((__m512*)md,0xF,x[0]);
   else if (rsiz == 104){
   	_mm512_mask_storeu_epi64((__m512*)md,0x1F,x[0]);
	 md+=40;
    	_mm512_mask_storeu_epi64((__m512*)md,0x01,x[1]);
   }else if(rsiz == 72){
	_mm512_mask_storeu_epi64((__m512*)md,0x1F,x[0]);
	 md+=40;
    	_mm512_mask_storeu_epi64((__m512*)md,0x0F,x[1]);
   }


    
    return 0;
}


