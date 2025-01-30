#include <stdint.h>
#include "params.h"
#include "rounding.h"

/*************************************************
* Name:        power2round
*
* Description: For finite field element a, compute a0, a1 such that
*              a mod^+ Q = a1*2^D + a0 with -2^{D-1} < a0 <= 2^{D-1}.
*              Assumes a to be standard representative.
*
* Arguments:   - int32_t a: input element
*              - int32_t *a0: pointer to output element a0
*
* Returns a1.
**************************************************/
int32_t power2round(int32_t *a0, int32_t a)  {
  int32_t a1;

  a1 = (a + (1 << (D-1)) - 1) >> D;
  *a0 = a - (a1 << D);
  return a1;
}
void power2round_avx(__m512i *a1, __m512i *a0, const __m512i *a)
{
  unsigned int i;
  __m512i f,f0,f1;
  const __m512i mask = _mm512_set1_epi32(-(1 << D));
  const __m512i half = _mm512_set1_epi32((1 << (D-1)) - 1);

  for(i = 0; i < N/16; ++i) {
    f = _mm512_load_si512(&a[i]);
    f1 = _mm512_add_epi32(f,half);
    f0 = _mm512_and_si512(f1,mask);
    f1 = _mm512_srli_epi32(f1,D);
    f0 = _mm512_sub_epi32(f,f0);
    _mm512_store_si512(&a1[i],f1);
    _mm512_store_si512(&a0[i],f0);
  }
}
/*************************************************
* Name:        decompose
*
* Description: For finite field element a, compute high and low bits a0, a1 such
*              that a mod^+ Q = a1*ALPHA + a0 with -ALPHA/2 < a0 <= ALPHA/2 except
*              if a1 = (Q-1)/ALPHA where we set a1 = 0 and
*              -ALPHA/2 <= a0 = a mod^+ Q - Q < 0. Assumes a to be standard
*              representative.
*
* Arguments:   - int32_t a: input element
*              - int32_t *a0: pointer to output element a0
*
* Returns a1.
**************************************************/
int32_t decompose(int32_t *a0, int32_t a) {
  int32_t a1;

  a1  = (a + 127) >> 7;
#if GAMMA2 == (Q-1)/32
  a1  = (a1*1025 + (1 << 21)) >> 22;
  a1 &= 15;
#elif GAMMA2 == (Q-1)/88
  a1  = (a1*11275 + (1 << 23)) >> 24;
  a1 ^= ((43 - a1) >> 31) & a1;
#endif

  *a0  = a - a1*2*GAMMA2;
  *a0 -= (((Q-1)/2 - *a0) >> 31) & Q;
  return a1;
}

#if GAMMA2 == (Q-1)/32
void decompose_avx(__m512i *a1, __m512i *a0, const __m512i *a)
{
  unsigned int i;
  __m512i f,f0,f1,g2;
  __mmask16 good0;
  const __m512i q = _mm512_set1_epi32(Q);
  const __m512i v = _mm512_set1_epi32(1025);
  const __m512i alpha = _mm512_set1_epi32(2*GAMMA2);
  const __m512i zero = _mm512_setzero_si512();
  const __m512i off = _mm512_set1_epi32(127);
  const __m512i shift2 = _mm512_set1_epi32(2097152);
  const __m512i mask = _mm512_set1_epi32(15);
  const __m512i qsubone = _mm512_set1_epi32(4190208);
  for(i=0;i<N/16;i++) {
    f = _mm512_load_si512(&a[i]);
    f1 = _mm512_add_epi32(f,off);
    f1 = _mm512_srli_epi32(f1,7);
    f1 = _mm512_mullo_epi32(f1,v);
    f1 = _mm512_add_epi32(f1,shift2);
    f1 = _mm512_srli_epi32(f1,22);
    f1 = _mm512_and_si512(f1,mask);
    f0 = _mm512_mullo_epi32(f1,alpha);
    f0 = _mm512_sub_epi32(f,f0);
    good0 = _mm512_cmp_epi32_mask(f0, qsubone, 6);
    g2 = _mm512_mask_blend_epi32(good0, zero, q);
    f0 = _mm512_sub_epi32(f0,g2);
    _mm512_store_si512(&a1[i],f1);
    _mm512_store_si512(&a0[i],f0);
  }
}

#elif GAMMA2 == (Q-1)/88
void decompose_avx(__m512i *a1, __m512i *a0, const __m512i *a)
{
  unsigned int i;
  __m512i f,f0,f1,f2,t,g2;
  __mmask16 good0;
  const __m512i q = _mm512_set1_epi32(Q);
  const __m512i zero = _mm512_setzero_si512();
  const __m512i v = _mm512_set1_epi32(11275);
  const __m512i v2 = _mm512_set1_epi32(44);
  const __m512i alpha = _mm512_set1_epi32(2*GAMMA2);
  const __m512i off = _mm512_set1_epi32(127);
  const __m512i max = _mm512_set1_epi32(43);
  const __m512i shift2 = _mm512_set1_epi32(8388608);
  const __m512i qsubone = _mm512_set1_epi32(4190208);
  for(i=0;i<N/16;i++) {
    f = _mm512_load_si512(&a[i]);
    f1 = _mm512_add_epi32(f,off);
    f1 = _mm512_srli_epi32(f1,7);
    f1 = _mm512_mullo_epi32(f1,v);
    f1 = _mm512_add_epi32(f1,shift2);
    f1 = _mm512_srli_epi32(f1,24);
    t = _mm512_sub_epi32(max,f1);
    t = _mm512_srli_epi32(t,31);
    t = _mm512_and_epi32(t,f1);
    f2 = _mm512_xor_epi32(t,f1);
    good0 = _mm512_cmp_epi32_mask(f2, v2, 0);
    f2 = _mm512_mask_blend_epi32(good0, f2, zero);
    f0 = _mm512_mullo_epi32(f2,alpha);
    f0 = _mm512_sub_epi32(f,f0);
    good0 = _mm512_cmp_epi32_mask(f0, qsubone, 6);
    g2 = _mm512_mask_blend_epi32(good0, zero, q);
    f0 = _mm512_sub_epi32(f0,g2);
    _mm512_store_si512(&a1[i],f2);
    _mm512_store_si512(&a0[i],f0);
  }
}
#endif
/*************************************************
* Name:        make_hint
*
* Description: Compute hint bit indicating whether the low bits of the
*              input element overflow into the high bits. Inputs assumed
*              to be standard representatives.
*
* Arguments:   - int32_t a0: low bits of input element
*              - int32_t a1: high bits of input element
*
* Returns 1 if overflow.
**************************************************/
unsigned int make_hint(int32_t a0, int32_t a1) {
  if(a0 <= GAMMA2 || a0 > Q - GAMMA2 || (a0 == Q - GAMMA2 && a1 == 0))
    return 0;

  return 1;
}
unsigned int make_hint_avx(int32_t * restrict h, const int32_t * restrict a0, const int32_t * restrict a1)
{
  unsigned int i, n = 0;
  __m512i f0, f1;
  __mmask16 g0, g1, g2;
  const __m512i one = _mm512_set1_epi32(1);
  const __m512i bound1 = _mm512_set1_epi32(GAMMA2+1);
  const __m512i bound2 = _mm512_set1_epi32(Q-GAMMA2);
  const __m512i zero = _mm512_setzero_si512();
  for(i = 0; i < N/16; ++i) {
    f0 = _mm512_load_si512((__m512i *)&a0[16*i]);
    f1 = _mm512_load_si512((__m512i *)&a1[16*i]);

    g0 = _mm512_cmpgt_epi32_mask(bound1,f0);
    g1 = _mm512_cmpgt_epi32_mask(f0,bound2);

    g0 |= g1;
    g1 = _mm512_cmpeq_epi32_mask(f0,bound2);
    g2 = _mm512_cmpeq_epi32_mask(f1,zero);
    g1 &= g2;
    g0 |= g1;
    n += _mm_popcnt_u32(g0);
    f0 = _mm512_mask_set1_epi32(one, g0, 0);
    _mm512_store_si512((__m512i *)&h[16*i],f0);
  }
  return N - n;
}
/*************************************************
* Name:        use_hint
*
* Description: Correct high bits according to hint.
*
* Arguments:   - int32_t a: input element
*              - unsigned int hint: hint bit
*
* Returns corrected high bits.
**************************************************/
int32_t use_hint(int32_t a, unsigned int hint) {
  int32_t a0, a1;

  a1 = decompose(&a0, a);
  if(hint == 0)
    return a1;

#if GAMMA2 == (Q-1)/32
  if(a0 > 0)
    return (a1 + 1) & 15;
  else
    return (a1 - 1) & 15;
#elif GAMMA2 == (Q-1)/88
  if(a0 > 0)
    return (a1 == 43) ?  0 : a1 + 1;
  else
    return (a1 ==  0) ? 43 : a1 - 1;
#endif
}
void use_hint_avx(__m512i *b, const __m512i *a, const __m512i * restrict hint) {
  unsigned int i;
  __m512i a0[N/16];
  __m512i f,g,h,g0,g1,f2,f3;
  __mmask16 good0,good3;
  const __m512i one = _mm512_set1_epi32(1);
  const __m512i zero = _mm512_setzero_si512();
#if GAMMA2 == (Q-1)/32
  const __m512i mask = _mm512_set1_epi32(15);
#elif GAMMA2 == (Q-1)/88
  const __m512i max = _mm512_set1_epi32(43);
#endif

  decompose_avx(b, a0, a);
  
  for(i=0;i<N/16;i++) {
    f = _mm512_load_si512(&a0[i]);
    g = _mm512_load_si512(&b[i]);
    h = _mm512_load_si512(&hint[i]);
    f2 = _mm512_add_epi32(g, one);
    f3 = _mm512_sub_epi32(g, one);
    good3 = _mm512_cmp_epi32_mask(f, zero, 6);
    #if GAMMA2 == (Q-1)/32
      __m512i f1, f0;
      f0 = _mm512_and_si512(f2, mask);
      f1 = _mm512_and_si512(f3, mask); 
      g0 = _mm512_mask_blend_epi32(good3, f1, f0);
    #elif GAMMA2 == (Q-1)/88
      __mmask16 good1;
      good0 = _mm512_cmp_epi32_mask(g, max, 0);
      g0 = _mm512_mask_blend_epi32(good0, f2, zero);
      good1 = _mm512_cmp_epi32_mask(g, zero, 0);
      g1 = _mm512_mask_blend_epi32(good1, f3, max);
      g0 = _mm512_mask_blend_epi32(good3, g1, g0);
    #endif
    good0 = _mm512_cmp_epi32_mask(h, zero, 0);
    g1 = _mm512_mask_blend_epi32(good0, g0, g);
    _mm512_store_si512(&b[i],g1);
  }

}