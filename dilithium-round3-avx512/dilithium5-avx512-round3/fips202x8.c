#include <immintrin.h>
#include <stdint.h>
#include <assert.h>
#include "fips202x8.h"
#include "fips202.h"
#include "SHA-3_256par.h"

#define NROUNDS 24
#define ROL(a, offset) ((a << offset) ^ (a >> (64-offset)))



/* Use implementation from the Keccak Code Package */
extern void KeccakP1600times8_PermuteAll_24rounds(__m512i *s);
#define KeccakF1600_StatePermute8x KeccakP1600times8_PermuteAll_24rounds

static void keccak_absorb8x(__m512i *s,
                          unsigned int r,
                          const unsigned char *m0,
                          const unsigned char *m1,
                          const unsigned char *m2,
                          const unsigned char *m3,
                          const unsigned char *m4,
                          const unsigned char *m5,
                          const unsigned char *m6,
                          const unsigned char *m7,
                          unsigned long long int mlen,
                          unsigned char p)
{
  unsigned long long i;
  uint64_t pos = 0;
  __m512i idx, t;
  idx = _mm512_set_epi64((long long)m7, (long long)m6, (long long)m5, (long long)m4,(long long)m3, (long long)m2, (long long)m1, (long long)m0);

  while (mlen >= r)
  {
    for (i = 0; i < r / 8; ++i)
    {
      t = _mm512_i64gather_epi64(idx, (long long *)pos, 1);
      s[i] = _mm512_xor_si512(s[i], t);
      pos += 8;
    }
    keccakF8x(s);
    mlen -= r;
  }


  for(i = 0; i < mlen/8; ++i) {
    t = _mm512_i64gather_epi64(idx, (long long *)pos, 1);
    s[i] = _mm512_xor_si512(s[i], t);
    pos += 8;
  }
  mlen -= 8*i;


  if(mlen) {
    t = _mm512_i64gather_epi64(idx, (long long *)pos, 1);
    idx = _mm512_set1_epi64((1ULL << (8*mlen)) - 1);
    t = _mm512_and_si512(t, idx);
    s[i] = _mm512_xor_si512(s[i], t);
  }
  t = _mm512_set1_epi64((uint64_t)p << 8*mlen);
  s[i] = _mm512_xor_si512(s[i], t);
  t = _mm512_set1_epi64(1ULL << 63);
  s[r/8 - 1] = _mm512_xor_si512(s[r/8 - 1], t);
  
}


void keccak_squeezeblocks8x(unsigned char *h0,
                                   unsigned char *h1,
                                   unsigned char *h2,
                                   unsigned char *h3,
                                   unsigned char *h4,
                                   unsigned char *h5,
                                   unsigned char *h6,
                                   unsigned char *h7,
                                   unsigned long long int nblocks,
                                   __m512i *s,
                                   unsigned int r)
{
  unsigned int i;

  __m128d t;
  while(nblocks > 0)
  {
    keccakF8x(s);
    for(i=0;i<(r>>3);i++)
    {
      t = _mm_castsi128_pd(_mm512_castsi512_si128(s[i]));
      _mm_storel_pd((__attribute__((__may_alias__)) double *)&h0[8*i], t);
      _mm_storeh_pd((__attribute__((__may_alias__)) double *)&h1[8*i], t);
      t = _mm_castsi128_pd(_mm512_extracti64x2_epi64(s[i],1));
      _mm_storel_pd((__attribute__((__may_alias__)) double *)&h2[8*i], t);
      _mm_storeh_pd((__attribute__((__may_alias__)) double *)&h3[8*i], t);
      t = _mm_castsi128_pd(_mm512_extracti64x2_epi64(s[i],2));
      _mm_storel_pd((__attribute__((__may_alias__)) double *)&h4[8*i], t);
      _mm_storeh_pd((__attribute__((__may_alias__)) double *)&h5[8*i], t);
      t = _mm_castsi128_pd(_mm512_extracti64x2_epi64(s[i],3));
      _mm_storel_pd((__attribute__((__may_alias__)) double *)&h6[8*i], t);
      _mm_storeh_pd((__attribute__((__may_alias__)) double *)&h7[8*i], t);

    }
    h0 += r;
    h1 += r;
    h2 += r;
    h3 += r;
    h4 += r;
    h5 += r;
    h6 += r;
    h7 += r;
    nblocks--;
  }
}



void shake128x8(__m512i s[25], unsigned char *out0,
                unsigned char *out1,
                unsigned char *out2,
                unsigned char *out3,
                unsigned char *out4,
                unsigned char *out5,
                unsigned char *out6,
                unsigned char *out7, unsigned long long outlen,
                unsigned char *in0,
                unsigned char *in1,
                unsigned char *in2,
                unsigned char *in3,
                unsigned char *in4,
                unsigned char *in5,
                unsigned char *in6,
                unsigned char *in7, unsigned long long inlen)
{
  unsigned char t0[SHAKE128_RATE];
  unsigned char t1[SHAKE128_RATE];
  unsigned char t2[SHAKE128_RATE];
  unsigned char t3[SHAKE128_RATE];
  unsigned char t4[SHAKE128_RATE];
  unsigned char t5[SHAKE128_RATE];
  unsigned char t6[SHAKE128_RATE];
  unsigned char t7[SHAKE128_RATE];
  unsigned int i;

  /* zero state */
  for(i=0;i<25;i++)
    s[i] = _mm512_xor_si512(s[i], s[i]);

  /* absorb 8 message of identical length in parallel */
  keccak_absorb8x(s, SHAKE128_RATE, in0, in1, in2, in3, in4, in5, in6, in7, inlen, 0x1F);

  /* Squeeze output */
  keccak_squeezeblocks8x(out0, out1, out2, out3, out4, out5, out6, out7, outlen/SHAKE128_RATE, s, SHAKE128_RATE);

  out0 += (outlen/SHAKE128_RATE)*SHAKE128_RATE;
  out1 += (outlen/SHAKE128_RATE)*SHAKE128_RATE;
  out2 += (outlen/SHAKE128_RATE)*SHAKE128_RATE;
  out3 += (outlen/SHAKE128_RATE)*SHAKE128_RATE;
  out4 += (outlen/SHAKE128_RATE)*SHAKE128_RATE;
  out5 += (outlen/SHAKE128_RATE)*SHAKE128_RATE;
  out6 += (outlen/SHAKE128_RATE)*SHAKE128_RATE;
  out7 += (outlen/SHAKE128_RATE)*SHAKE128_RATE;

  if(outlen%SHAKE128_RATE)
  {
    keccak_squeezeblocks8x(t0, t1, t2, t3, t4, t5, t6, t7, 1, s, SHAKE128_RATE);
    for(i=0;i<outlen%SHAKE128_RATE;i++)
    {
      out0[i] = t0[i];
      out1[i] = t1[i];
      out2[i] = t2[i];
      out3[i] = t3[i];
      out4[i] = t4[i];
      out5[i] = t5[i];
      out6[i] = t6[i];
      out7[i] = t7[i];
    }
  }
}


void shake256x8(__m512i s[25], unsigned char *out0,
                unsigned char *out1,
                unsigned char *out2,
                unsigned char *out3,
                unsigned char *out4,
                unsigned char *out5,
                unsigned char *out6,
                unsigned char *out7, unsigned long long outlen,
                unsigned char *in0,
                unsigned char *in1,
                unsigned char *in2,
                unsigned char *in3,
                unsigned char *in4,
                unsigned char *in5,
                unsigned char *in6,
                unsigned char *in7, unsigned long long inlen)
{ 
  unsigned char t0[SHAKE256_RATE];
  unsigned char t1[SHAKE256_RATE];
  unsigned char t2[SHAKE256_RATE];
  unsigned char t3[SHAKE256_RATE];
  unsigned char t4[SHAKE256_RATE];
  unsigned char t5[SHAKE256_RATE];
  unsigned char t6[SHAKE256_RATE];
  unsigned char t7[SHAKE256_RATE];
  unsigned int i;

  /* zero state */
  for(i=0;i<25;i++)
    s[i] = _mm512_xor_si512(s[i], s[i]);

  /* absorb 8 message of identical length in parallel */
  keccak_absorb8x(s, SHAKE256_RATE, in0, in1, in2, in3, in4, in5, in6, in7, inlen, 0x1F);

  /* Squeeze output */
  keccak_squeezeblocks8x(out0, out1, out2, out3,out4, out5, out6, out7, outlen/SHAKE256_RATE, s, SHAKE256_RATE);

  out0 += (outlen/SHAKE256_RATE)*SHAKE256_RATE;
  out1 += (outlen/SHAKE256_RATE)*SHAKE256_RATE;
  out2 += (outlen/SHAKE256_RATE)*SHAKE256_RATE;
  out3 += (outlen/SHAKE256_RATE)*SHAKE256_RATE;
  out4 += (outlen/SHAKE256_RATE)*SHAKE256_RATE;
  out5 += (outlen/SHAKE256_RATE)*SHAKE256_RATE;
  out6 += (outlen/SHAKE256_RATE)*SHAKE256_RATE;
  out7 += (outlen/SHAKE256_RATE)*SHAKE256_RATE;

  if(outlen%SHAKE256_RATE)
  {
    keccak_squeezeblocks8x(t0, t1, t2, t3, t4, t5, t6, t7, 1, s, SHAKE256_RATE);
    for(i=0;i<outlen%SHAKE256_RATE;i++)
    {
      out0[i] = t0[i];
      out1[i] = t1[i];
      out2[i] = t2[i];
      out3[i] = t3[i];
      out4[i] = t4[i];
      out5[i] = t5[i];
      out6[i] = t6[i];
      out7[i] = t7[i];

    }
  }
}
