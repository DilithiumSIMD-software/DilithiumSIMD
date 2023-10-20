#include <stdint.h>
#include <immintrin.h>
#include "params.h"
#include "rejsample.h"
#include "symmetric.h"
#include "shuffle8.h"

unsigned int rej_uniform_avx(int32_t * restrict r, const uint8_t buf[REJ_UNIFORM_BUFLEN+8])
{
  unsigned int ctr, pos;
  __mmask16 good;
  __m512i d;
  const __m512i bound = _mm512_set1_epi32(Q);
  const __m512i zero = _mm512_setzero_si512();
  const __m512i mask  = _mm512_set1_epi32(0x7FFFFF);

  const __m512i idx8  = _mm512_set_epi8(48,47,46,45,48,44,43,42,
                                        48,41,40,39,48,38,37,36,
                                        48,35,34,33,48,32,31,30,
                                        48,29,28,27,48,26,25,24,
                                        48,23,22,21,48,20,19,18,
                                        48,17,16,15,48,14,13,12,
                                        48,11,10, 9,48, 8, 7, 6,
                                        48, 5, 4, 3,48, 2, 1, 0);
  ctr = pos = 0;
  while(pos <= REJ_UNIFORM_BUFLEN - 48) {
    d = _mm512_loadu_si512((__m512i *)&buf[pos]);
    d = _mm512_mask_blend_epi64(0x3F, zero, d);
    d = _mm512_permutexvar_epi8(idx8, d);
    d = _mm512_and_si512(d, mask);
    pos += 48;
    good = _mm512_cmp_epi32_mask(d, bound, 1);
    _mm512_mask_compressstoreu_epi32(&r[ctr], good, d);
    ctr += _mm_popcnt_u32((int32_t)good);
    if(ctr > N - 8) break;
  }
  uint32_t t;
  while(ctr < N && pos <= REJ_UNIFORM_BUFLEN - 3) {
    t  = buf[pos++];
    t |= (uint32_t)buf[pos++] << 8;
    t |= (uint32_t)buf[pos++] << 16;
    t &= 0x7FFFFF;
    if(t < Q)
      r[ctr++] = t;
  }
  return ctr;
}

#if ETA == 2
unsigned int rej_eta_avx(int32_t * restrict r, const uint8_t buf[REJ_UNIFORM_ETA_BUFLEN]) {
  unsigned int ctr, pos;
  __mmask16 good0, good1;
  __m512i f0, f1, f2, f3, f4, f5;
  const __m512i mask = _mm512_set1_epi32(15);
  const __m512i eta = _mm512_set1_epi32(ETA);
  const __m512i bound = mask;
  const __m512i idx16  = _mm512_set_epi32(15, 7, 14, 6,13, 5,12, 4,
                                          11, 3, 10, 2, 9, 1, 8, 0);
  const __m512i v = _mm512_set1_epi32(205);
  const __m512i p = _mm512_set1_epi32(5);
  ctr = pos = 0;
  while(ctr <= N - 32 && pos <= REJ_UNIFORM_ETA_BUFLEN - 16) {
    f0 = _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i *)&buf[pos]));
    f1 = _mm512_srai_epi32(f0,4);
    f0 = _mm512_and_si512(f0,mask);
    // to get the right order
    shufflelo8_avx(&f4, &f0, &f1);
    shufflehi8_avx(&f5, &f0, &f1);
    f0 = _mm512_permutexvar_epi32(idx16, f4);
    f1 = _mm512_permutexvar_epi32(idx16, f5);
    good0 = _mm512_cmp_epi32_mask(f0, bound, 1);
    good1 = _mm512_cmp_epi32_mask(f1, bound, 1);
    f2 = _mm512_mullo_epi32(f0,v);
    f2 = _mm512_srai_epi32(f2,10);
    f2 = _mm512_mullo_epi32(f2,p);
    f0 = _mm512_sub_epi32(f0,f2);
    f0 = _mm512_sub_epi32(eta,f0);
    //store
    _mm512_mask_compressstoreu_epi32(&r[ctr], good0, f0);
    ctr += _mm_popcnt_u32((int32_t)good0);
    f3 = _mm512_mullo_epi32(f1,v);
    f3 = _mm512_srai_epi32(f3,10);
    f3 = _mm512_mullo_epi32(f3,p);
    f1 = _mm512_sub_epi32(f1,f3);
    f1 = _mm512_sub_epi32(eta,f1);
    _mm512_mask_compressstoreu_epi32(&r[ctr], good1, f1);
    pos += 16;
    ctr += _mm_popcnt_u32((int32_t)good1);
    if(ctr > N - 32) break;
  }  
  uint32_t t0, t1;
  while(ctr < N && pos < REJ_UNIFORM_ETA_BUFLEN) {
    t0 = buf[pos] & 0x0F;
    t1 = buf[pos++] >> 4;

    if(t0 < 15) {
      t0 = t0 - (205*t0 >> 10)*5;
      r[ctr++] = 2 - t0;
    }
    if(t1 < 15 && ctr < N) {
      t1 = t1 - (205*t1 >> 10)*5;
      r[ctr++] = 2 - t1;
    }
  }

  return ctr;
}
#elif ETA == 4
unsigned int rej_eta_avx(int32_t * restrict r, const uint8_t buf[REJ_UNIFORM_ETA_BUFLEN]) {
  unsigned int ctr, pos;
  __mmask16 good0, good1;
  __m512i f0, f1, f2, f3;
  const __m512i mask = _mm512_set1_epi32(15);
  const __m512i eta = _mm512_set1_epi32(ETA);
  const __m512i bound = _mm512_set1_epi32(9);
  const __m512i idx16  = _mm512_set_epi32(15, 7, 14, 6,13, 5,12, 4,
                                          11, 3, 10, 2, 9, 1, 8, 0);
  ctr = pos = 0;
  while(ctr <= N - 32 && pos <= REJ_UNIFORM_ETA_BUFLEN - 16) {
    f0 = _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i *)&buf[pos]));
    f1 = _mm512_srai_epi32(f0,4);
    f0 = _mm512_and_si512(f0,mask);
    // to get the right order
    f2 = _mm512_shuffle_i32x4(f0, f1, 0x44);
    f3 = _mm512_shuffle_i32x4(f0, f1, 0xEE);
    f0 = _mm512_permutexvar_epi32(idx16, f2);
    f1 = _mm512_permutexvar_epi32(idx16, f3);
    good0 = _mm512_cmp_epi32_mask(f0, bound, 1);
    good1 = _mm512_cmp_epi32_mask(f1, bound, 1);
    //store
    f0 = _mm512_sub_epi32(eta,f0);
    _mm512_mask_compressstoreu_epi32(&r[ctr], good0, f0);
    ctr += _mm_popcnt_u32((int32_t)good0);
    
    f1 = _mm512_sub_epi32(eta,f1);  
    _mm512_mask_compressstoreu_epi32(&r[ctr], good1, f1);
    ctr += _mm_popcnt_u32((int32_t)good1);
    pos += 16;
    if(ctr > N - 32) break;
  }  
  uint32_t t0, t1;
  while(ctr < N && pos < REJ_UNIFORM_ETA_BUFLEN) {
    t0 = buf[pos] & 0x0F;
    t1 = buf[pos++] >> 4;

    if(t0 < 9)
      r[ctr++] = 4 - t0;
    if(t1 < 9 && ctr < N)
      r[ctr++] = 4 - t1;
  }

  return ctr;
}
#endif
