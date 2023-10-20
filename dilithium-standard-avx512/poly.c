#include <stdint.h>
#include "params.h"
#include "poly.h"
#include "ntt.h"
#include "reduce.h"
#include "rounding.h"
#include "symmetric.h"
#include "consts.h"
#include "fips202x8.h"
#include "rejsample.h"
#ifdef DBENCH
#include "test/cpucycles.h"
extern const uint64_t timing_overhead;
extern uint64_t *tred, *tadd, *tmul, *tround, *tsample, *tpack;
#define DBENCH_START() uint64_t time = cpucycles()
#define DBENCH_STOP(t) t += cpucycles() - time - timing_overhead
#else
#define DBENCH_START()
#define DBENCH_STOP(t)
#endif
void poly_nttunpack(poly *a) {
  DBENCH_START();

  nttunpack_avx(a->vec);

  DBENCH_STOP(*tmul);
}
/*************************************************
* Name:        poly_reduce
*
* Description: Inplace reduction of all coefficients of polynomial to
*              representative in [-6283009,6283007].
*
* Arguments:   - poly *a: pointer to input/output polynomial
**************************************************/
// void poly_reduce(poly *a) {
//   unsigned int i;
//   DBENCH_START();

//   for(i = 0; i < N; ++i)
//     a->coeffs[i] = reduce32(a->coeffs[i]);

//   DBENCH_STOP(*tred);
// }
void poly_reduce(poly *a) {
  unsigned int i;
  __m512i f,g;
  const __m512i q = _mm512_set1_epi32(Q);
  const __m512i off = _mm512_set1_epi32(4194304);


  for(i = 0; i < N/16; i++) {
    f = _mm512_load_si512(&a->vec[i]);
    g = _mm512_add_epi32(f,off);
    g = _mm512_srai_epi32(g,23);
    g = _mm512_mullo_epi32(g,q);
    f = _mm512_sub_epi32(f,g);
    _mm512_store_si512(&a->vec[i],f);
  }

}
/*************************************************
* Name:        poly_caddq
*
* Description: For all coefficients of in/out polynomial add Q if
*              coefficient is negative.
*
* Arguments:   - poly *a: pointer to input/output polynomial
**************************************************/
// void poly_caddq(poly *a) {
//   unsigned int i;
//   DBENCH_START();

//   for(i = 0; i < N; ++i)
//     a->coeffs[i] = caddq(a->coeffs[i]);

//   DBENCH_STOP(*tred);
// }
void poly_caddq(poly *a) {
  unsigned int i;
  __m512i f,g;
  const __m512i q = _mm512_set1_epi32(Q);
  for(i = 0; i < N/16; i++) {
    f = _mm512_load_si512(&a->vec[i]);
    g = _mm512_srai_epi32(f, 31);
    g = _mm512_and_epi32(q,g);
    f = _mm512_add_epi32(f,g);
    _mm512_store_si512(&a->vec[i],f);
  }

}
/*************************************************
* Name:        poly_add
*
* Description: Add polynomials. No modular reduction is performed.
*
* Arguments:   - poly *c: pointer to output polynomial
*              - const poly *a: pointer to first summand
*              - const poly *b: pointer to second summand
**************************************************/
// void poly_add(poly *c, const poly *a, const poly *b)  {
//   unsigned int i;
//   DBENCH_START();

//   for(i = 0; i < N; ++i)
//     c->coeffs[i] = a->coeffs[i] + b->coeffs[i];

//   DBENCH_STOP(*tadd);
// }
void poly_add(poly *c, const poly *a, const poly *b)  {
  unsigned int i;
  __m512i f,g;

  for(i = 0; i < N/16; i++) {
    f = _mm512_load_si512(&a->vec[i]);
    g = _mm512_load_si512(&b->vec[i]);
    f = _mm512_add_epi32(f,g);
    _mm512_store_si512(&c->vec[i],f);
  }
}

/*************************************************
* Name:        poly_sub
*
* Description: Subtract polynomials. No modular reduction is
*              performed.
*
* Arguments:   - poly *c: pointer to output polynomial
*              - const poly *a: pointer to first input polynomial
*              - const poly *b: pointer to second input polynomial to be
*                               subtraced from first input polynomial
**************************************************/
// void poly_sub(poly *c, const poly *a, const poly *b) {
//   unsigned int i;
//   DBENCH_START();

//   for(i = 0; i < N; ++i)
//     c->coeffs[i] = a->coeffs[i] - b->coeffs[i];

//   DBENCH_STOP(*tadd);
// }
void poly_sub(poly *c, const poly *a, const poly *b) {
  unsigned int i;
  DBENCH_START();

  __m512i f,g;

  for(i = 0; i < N/16; i++) {
    f = _mm512_load_si512(&a->vec[i]);
    g = _mm512_load_si512(&b->vec[i]);
    f = _mm512_sub_epi32(f,g);
    _mm512_store_si512(&c->vec[i],f);
  }


  DBENCH_STOP(*tadd);
}

/*************************************************
* Name:        poly_shiftl
*
* Description: Multiply polynomial by 2^D without modular reduction. Assumes
*              input coefficients to be less than 2^{31-D} in absolute value.
*
* Arguments:   - poly *a: pointer to input/output polynomial
**************************************************/
// void poly_shiftl(poly *a) {
//   unsigned int i;
//   DBENCH_START();

//   for(i = 0; i < N; ++i)
//     a->coeffs[i] <<= D;

//   DBENCH_STOP(*tmul);
// }
void poly_shiftl(poly *a) {
  unsigned int i;
  DBENCH_START();

   __m512i f;

  for(i = 0; i < N/16; i++) {
    f = _mm512_load_si512(&a->vec[i]);
    f = _mm512_slli_epi32(f,D);
    _mm512_store_si512(&a->vec[i],f);
  }

  DBENCH_STOP(*tmul);
}

/*************************************************
* Name:        poly_ntt
*
* Description: Inplace forward NTT. Coefficients can grow by
*              8*Q in absolute value.
*
* Arguments:   - poly *a: pointer to input/output polynomial
**************************************************/
// void poly_ntt(poly *a) {
//   DBENCH_START();

//   ntt(a->coeffs);

//   DBENCH_STOP(*tmul);
// }
void poly_ntt(poly *a) {
  DBENCH_START();

  // ntt(a->coeffs);
  ntt_avx(a->vec, qdata.vec);

  DBENCH_STOP(*tmul);
}

void poly_smallntt(poly *a) {
  DBENCH_START();

  smallntt_avx(a->vec, qdata.vec);

  DBENCH_STOP(*tmul);
}
void poly_instailoredntt(poly *a) {
  DBENCH_START();

  instailoredntt_avx(a->vec, qdata.vec);

  DBENCH_STOP(*tmul);
}
/*************************************************
* Name:        poly_invntt_tomont
*
* Description: Inplace inverse NTT and multiplication by 2^{32}.
*              Input coefficients need to be less than Q in absolute
*              value and output coefficients are again bounded by Q.
*
* Arguments:   - poly *a: pointer to input/output polynomial
**************************************************/
// void poly_invntt_tomont(poly *a) {
//   DBENCH_START();

//   invntt_tomont(a->coeffs);

//   DBENCH_STOP(*tmul);
// }
void poly_invntt_tomont(poly *a) {
  DBENCH_START();

  // invntt_tomont(a->coeffs);
   invntt_avx(a->vec, qdata.vec);

  DBENCH_STOP(*tmul);
}

/*************************************************
* Name:        poly_pointwise_montgomery
*
* Description: Pointwise multiplication of polynomials in NTT domain
*              representation and multiplication of resulting polynomial
*              by 2^{-32}.
*
* Arguments:   - poly *c: pointer to output polynomial
*              - const poly *a: pointer to first input polynomial
*              - const poly *b: pointer to second input polynomial
**************************************************/
// void poly_pointwise_montgomery(poly *c, const poly *a, const poly *b) {
//   unsigned int i;
//   DBENCH_START();

//   for(i = 0; i < N; ++i)
//     c->coeffs[i] = montgomery_reduce((int64_t)a->coeffs[i] * b->coeffs[i]);

//   DBENCH_STOP(*tmul);
// }
void poly_pointwise_montgomery(poly *c, const poly *a, const poly *b) {
  // unsigned int i;
  DBENCH_START();
  pointwise_avx(c->vec, a->vec, b->vec, qdata.vec);
  // for(i = 0; i < N; ++i)
    // c->coeffs[i] = montgomery_reduce((int64_t)a->coeffs[i] * b->coeffs[i]);
    

  DBENCH_STOP(*tmul);
}

/*************************************************
* Name:        poly_power2round
*
* Description: For all coefficients c of the input polynomial,
*              compute c0, c1 such that c mod Q = c1*2^D + c0
*              with -2^{D-1} < c0 <= 2^{D-1}. Assumes coefficients to be
*              standard representatives.
*
* Arguments:   - poly *a1: pointer to output polynomial with coefficients c1
*              - poly *a0: pointer to output polynomial with coefficients c0
*              - const poly *a: pointer to input polynomial
**************************************************/
// void poly_power2round(poly *a1, poly *a0, const poly *a) {
//   unsigned int i;
//   DBENCH_START();

//   for(i = 0; i < N; ++i)
//     a1->coeffs[i] = power2round(&a0->coeffs[i], a->coeffs[i]);

//   DBENCH_STOP(*tround);
// }
void poly_power2round(poly *a1, poly *a0, const poly *a) {
  // unsigned int i;
  DBENCH_START();

  // for(i = 0; i < N; ++i)
  //   a1->coeffs[i] = power2round(&a0->coeffs[i], a->coeffs[i]);
  power2round_avx(a1->vec, a0->vec, a->vec);

  DBENCH_STOP(*tround);
}

/*************************************************
* Name:        poly_decompose
*
* Description: For all coefficients c of the input polynomial,
*              compute high and low bits c0, c1 such c mod Q = c1*ALPHA + c0
*              with -ALPHA/2 < c0 <= ALPHA/2 except c1 = (Q-1)/ALPHA where we
*              set c1 = 0 and -ALPHA/2 <= c0 = c mod Q - Q < 0.
*              Assumes coefficients to be standard representatives.
*
* Arguments:   - poly *a1: pointer to output polynomial with coefficients c1
*              - poly *a0: pointer to output polynomial with coefficients c0
*              - const poly *a: pointer to input polynomial
**************************************************/
// void poly_decompose(poly *a1, poly *a0, const poly *a) {
//   unsigned int i;
//   DBENCH_START();

//   for(i = 0; i < N; ++i)
//     a1->coeffs[i] = decompose(&a0->coeffs[i], a->coeffs[i]);

//   DBENCH_STOP(*tround);
// }
void poly_decompose(poly *a1, poly *a0, const poly *a) {
  // unsigned int i;
  // DBENCH_START();

  // for(i = 0; i < N; ++i)
  //   a1->coeffs[i] = decompose(&a0->coeffs[i], a->coeffs[i]);
  

  // DBENCH_STOP(*tround);
  decompose_avx(a1->vec, a0->vec, a->vec);
}

/*************************************************
* Name:        poly_make_hint
*
* Description: Compute hint polynomial. The coefficients of which indicate
*              whether the low bits of the corresponding coefficient of
*              the input polynomial overflow into the high bits.
*
* Arguments:   - poly *h: pointer to output hint polynomial
*              - const poly *a0: pointer to low part of input polynomial
*              - const poly *a1: pointer to high part of input polynomial
*
* Returns number of 1 bits.
**************************************************/
// unsigned int poly_make_hint(poly *h, const poly *a0, const poly *a1) {
//   unsigned int i, s = 0;
//   DBENCH_START();

//   for(i = 0; i < N; ++i) {
//     h->coeffs[i] = make_hint(a0->coeffs[i], a1->coeffs[i]);
//     s += h->coeffs[i];
//   }

//   DBENCH_STOP(*tround);
//   return s;
// }
unsigned int poly_make_hint(poly *h, const poly *a0, const poly *a1) {
  // unsigned int i, s = 0;
  // DBENCH_START();

  // for(i = 0; i < N; ++i) {
  //   h->coeffs[i] = make_hint(a0->coeffs[i], a1->coeffs[i]);
  //   s += h->coeffs[i];
  // }
 
  // return s;
  // DBENCH_STOP(*tround);
   unsigned int r;
  r = make_hint_avx(h->vec, a0->vec, a1->vec);
  return r;
  
}
/*************************************************
* Name:        poly_use_hint
*
* Description: Use hint polynomial to correct the high bits of a polynomial.
*
* Arguments:   - poly *b: pointer to output polynomial with corrected high bits
*              - const poly *a: pointer to input polynomial
*              - const poly *h: pointer to input hint polynomial
**************************************************/
// void poly_use_hint(poly *b, const poly *a, const poly *h) {
//   unsigned int i;
//   DBENCH_START();

//   for(i = 0; i < N; ++i)
//     b->coeffs[i] = use_hint(a->coeffs[i], h->coeffs[i]);

//   DBENCH_STOP(*tround);
// }
void poly_use_hint(poly *b, const poly *a, const poly *h) {
  // unsigned int i;
  // DBENCH_START();

  // for(i = 0; i < N; ++i)
  //   b->coeffs[i] = use_hint(a->coeffs[i], h->coeffs[i]);

  // DBENCH_STOP(*tround);
  use_hint_avx(b->vec, a->vec, h->vec);
}
/*************************************************
* Name:        poly_chknorm
*
* Description: Check infinity norm of polynomial against given bound.
*              Assumes input coefficients were reduced by reduce32().
*
* Arguments:   - const poly *a: pointer to polynomial
*              - int32_t B: norm bound
*
* Returns 0 if norm is strictly smaller than B <= (Q-1)/8 and 1 otherwise.
**************************************************/
// int poly_chknorm(const poly *a, int32_t B) {
//   unsigned int i;
//   int32_t t;
//   DBENCH_START();

//   if(B > (Q-1)/8)
//     return 1;

//   /* It is ok to leak which coefficient violates the bound since
//      the probability for each coefficient is independent of secret
//      data but we must not leak the sign of the centralized representative. */
//   for(i = 0; i < N; ++i) {
//     /* Absolute value */
//     t = a->coeffs[i] >> 31;
//     t = a->coeffs[i] - (t & 2*a->coeffs[i]);

//     if(t >= B) {
//       DBENCH_STOP(*tsample);
//       return 1;
//     }
//   }

//   DBENCH_STOP(*tsample);
//   return 0;
// }
int poly_chknorm(const poly *a, int32_t B) {
 unsigned int i;
  __mmask16 good;
  uint16_t good1;
  good1 = 0;
  __m512i f;
  const __m512i bound = _mm512_set1_epi32(B);

  if(B > (Q-1)/8)
    return 1;
  for(i = 0; i < N/16; i++) {
    f = _mm512_load_si512(&a->vec[i]);
    f = _mm512_abs_epi32(f);
 
    good = _mm512_cmp_epi32_mask(f, bound, 5);
    good1 |= (uint16_t)good; 
    if(good1)
    {
      return 1;
    }
  }
  return 0;
}
/*************************************************
* Name:        rej_uniform
*
* Description: Sample uniformly random coefficients in [0, Q-1] by
*              performing rejection sampling on array of random bytes.
*
* Arguments:   - int32_t *a: pointer to output array (allocated)
*              - unsigned int len: number of coefficients to be sampled
*              - const uint8_t *buf: array of random bytes
*              - unsigned int buflen: length of array of random bytes
*
* Returns number of sampled coefficients. Can be smaller than len if not enough
* random bytes were given.
**************************************************/
static unsigned int rej_uniform(int32_t *a,
                                unsigned int len,
                                const uint8_t *buf,
                                unsigned int buflen)
{
  unsigned int ctr, pos;
  uint32_t t;
  DBENCH_START();

  ctr = pos = 0;
  while(ctr < len && pos + 3 <= buflen) {
    t  = buf[pos++];
    t |= (uint32_t)buf[pos++] << 8;
    t |= (uint32_t)buf[pos++] << 16;
    t &= 0x7FFFFF;

    if(t < Q)
      a[ctr++] = t;
  }

  DBENCH_STOP(*tsample);
  return ctr;
}

/*************************************************
* Name:        poly_uniform
*
* Description: Sample polynomial with uniformly random coefficients
*              in [0,Q-1] by performing rejection sampling on the
*              output stream of SHAKE256(seed|nonce)
*
* Arguments:   - poly *a: pointer to output polynomial
*              - const uint8_t seed[]: byte array with seed of length SEEDBYTES
*              - uint16_t nonce: 2-byte nonce
**************************************************/
#define POLY_UNIFORM_NBLOCKS ((768 + STREAM128_BLOCKBYTES - 1)/STREAM128_BLOCKBYTES)
void poly_uniform(poly *a,
                  const uint8_t seed[SEEDBYTES],
                  uint16_t nonce)
{
  unsigned int i, ctr, off;
  unsigned int buflen = POLY_UNIFORM_NBLOCKS*STREAM128_BLOCKBYTES;
  uint8_t buf[POLY_UNIFORM_NBLOCKS*STREAM128_BLOCKBYTES + 2];
  stream128_state state;

  stream128_init(&state, seed, nonce);
  stream128_squeezeblocks(buf, POLY_UNIFORM_NBLOCKS, &state);

  ctr = rej_uniform(a->coeffs, N, buf, buflen);

  while(ctr < N) {
    off = buflen % 3;
    for(i = 0; i < off; ++i)
      buf[i] = buf[buflen - off + i];

    stream128_squeezeblocks(buf + off, 1, &state);
    buflen = STREAM128_BLOCKBYTES + off;
    ctr += rej_uniform(a->coeffs + ctr, N - ctr, buf, buflen);
  }
}
void poly_uniform_8x(poly *a0,
                     poly *a1,
                     poly *a2,
                     poly *a3,
                     poly *a4,
                     poly *a5,
                     poly *a6,
                     poly *a7,
                     const uint8_t seed[32],
                     uint16_t nonce0,
                     uint16_t nonce1,
                     uint16_t nonce2,
                     uint16_t nonce3,
                     uint16_t nonce4,
                     uint16_t nonce5,
                     uint16_t nonce6,
                     uint16_t nonce7)
{
  unsigned int ctr0, ctr1, ctr2, ctr3, ctr4, ctr5, ctr6, ctr7;
  ALIGNED_UINT8(REJ_UNIFORM_BUFLEN+8) buf[8];

  __m512i f;

  f = _mm512_loadu_si512((__m512i *)&seed[0]);
  _mm512_store_si512(&buf[0].vec[0],f);
  _mm512_store_si512(&buf[1].vec[0],f);
  _mm512_store_si512(&buf[2].vec[0],f);
  _mm512_store_si512(&buf[3].vec[0],f);
  _mm512_store_si512(&buf[4].vec[0],f);
  _mm512_store_si512(&buf[5].vec[0],f);
  _mm512_store_si512(&buf[6].vec[0],f);
  _mm512_store_si512(&buf[7].vec[0],f);

  buf[0].coeffs[32] = nonce0;
  buf[0].coeffs[33] = nonce0 >> 8;
  buf[1].coeffs[32] = nonce1;
  buf[1].coeffs[33] = nonce1 >> 8;
  buf[2].coeffs[32] = nonce2;
  buf[2].coeffs[33] = nonce2 >> 8;
  buf[3].coeffs[32] = nonce3;
  buf[3].coeffs[33] = nonce3 >> 8;
  buf[4].coeffs[32] = nonce4;
  buf[4].coeffs[33] = nonce4 >> 8;
  buf[5].coeffs[32] = nonce5;
  buf[5].coeffs[33] = nonce5 >> 8;
  buf[6].coeffs[32] = nonce6;
  buf[6].coeffs[33] = nonce6 >> 8;
  buf[7].coeffs[32] = nonce7;
  buf[7].coeffs[33] = nonce7 >> 8;

  __m512i state[25];
  shake128x8(state, buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs, buf[4].coeffs, buf[5].coeffs, buf[6].coeffs, buf[7].coeffs, REJ_UNIFORM_BUFLEN, buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs, buf[4].coeffs, buf[5].coeffs, buf[6].coeffs, buf[7].coeffs, 34); 
 

  ctr0 = rej_uniform_avx(a0->coeffs, buf[0].coeffs);
  ctr1 = rej_uniform_avx(a1->coeffs, buf[1].coeffs);
  ctr2 = rej_uniform_avx(a2->coeffs, buf[2].coeffs);
  ctr3 = rej_uniform_avx(a3->coeffs, buf[3].coeffs);
  ctr4 = rej_uniform_avx(a4->coeffs, buf[4].coeffs);
  ctr5 = rej_uniform_avx(a5->coeffs, buf[5].coeffs);
  ctr6 = rej_uniform_avx(a6->coeffs, buf[6].coeffs);
  ctr7 = rej_uniform_avx(a7->coeffs, buf[7].coeffs);

   while(ctr0 < N || ctr1 < N || ctr2 < N || ctr3 < N || ctr4 < N || ctr5 < N || ctr6 < N || ctr7 < N) {
    keccak_squeezeblocks8x(buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs, buf[4].coeffs, buf[5].coeffs, buf[6].coeffs, buf[7].coeffs, 1, state, SHAKE128_RATE);
    ctr0 += rej_uniform(a0->coeffs + ctr0, N - ctr0, buf[0].coeffs, SHAKE128_RATE);
    ctr1 += rej_uniform(a1->coeffs + ctr1, N - ctr1, buf[1].coeffs, SHAKE128_RATE);
    ctr2 += rej_uniform(a2->coeffs + ctr2, N - ctr2, buf[2].coeffs, SHAKE128_RATE);
    ctr3 += rej_uniform(a3->coeffs + ctr3, N - ctr3, buf[3].coeffs, SHAKE128_RATE);
    ctr4 += rej_uniform(a4->coeffs + ctr4, N - ctr4, buf[4].coeffs, SHAKE128_RATE);
    ctr5 += rej_uniform(a5->coeffs + ctr5, N - ctr5, buf[5].coeffs, SHAKE128_RATE);
    ctr6 += rej_uniform(a6->coeffs + ctr6, N - ctr6, buf[6].coeffs, SHAKE128_RATE);
    ctr7 += rej_uniform(a7->coeffs + ctr7, N - ctr7, buf[7].coeffs, SHAKE128_RATE);
  
  
  }
}
/*************************************************
* Name:        rej_eta
*
* Description: Sample uniformly random coefficients in [-ETA, ETA] by
*              performing rejection sampling on array of random bytes.
*
* Arguments:   - int32_t *a: pointer to output array (allocated)
*              - unsigned int len: number of coefficients to be sampled
*              - const uint8_t *buf: array of random bytes
*              - unsigned int buflen: length of array of random bytes
*
* Returns number of sampled coefficients. Can be smaller than len if not enough
* random bytes were given.
**************************************************/
static unsigned int rej_eta(int32_t *a,
                            unsigned int len,
                            const uint8_t *buf,
                            unsigned int buflen)
{
  unsigned int ctr, pos;
  uint32_t t0, t1;
  DBENCH_START();

  ctr = pos = 0;
  while(ctr < len && pos < buflen) {
    t0 = buf[pos] & 0x0F;
    t1 = buf[pos++] >> 4;

#if ETA == 2
    if(t0 < 15) {
      t0 = t0 - (205*t0 >> 10)*5;
      a[ctr++] = 2 - t0;
    }
    if(t1 < 15 && ctr < len) {
      t1 = t1 - (205*t1 >> 10)*5;
      a[ctr++] = 2 - t1;
    }
#elif ETA == 4
    if(t0 < 9)
      a[ctr++] = 4 - t0;
    if(t1 < 9 && ctr < len)
      a[ctr++] = 4 - t1;
#endif
  }

  DBENCH_STOP(*tsample);
  return ctr;
}

/*************************************************
* Name:        poly_uniform_eta
*
* Description: Sample polynomial with uniformly random coefficients
*              in [-ETA,ETA] by performing rejection sampling on the
*              output stream from SHAKE256(seed|nonce)
*
* Arguments:   - poly *a: pointer to output polynomial
*              - const uint8_t seed[]: byte array with seed of length CRHBYTES
*              - uint16_t nonce: 2-byte nonce
**************************************************/
#if ETA == 2
#define POLY_UNIFORM_ETA_NBLOCKS ((136 + STREAM256_BLOCKBYTES - 1)/STREAM256_BLOCKBYTES)
#elif ETA == 4
#define POLY_UNIFORM_ETA_NBLOCKS ((227 + STREAM256_BLOCKBYTES - 1)/STREAM256_BLOCKBYTES)
#endif
void poly_uniform_eta(poly *a,
                      const uint8_t seed[CRHBYTES],
                      uint16_t nonce)
{
  unsigned int ctr;
  unsigned int buflen = POLY_UNIFORM_ETA_NBLOCKS*STREAM256_BLOCKBYTES;
  uint8_t buf[POLY_UNIFORM_ETA_NBLOCKS*STREAM256_BLOCKBYTES];
  stream256_state state;

  stream256_init(&state, seed, nonce);
  stream256_squeezeblocks(buf, POLY_UNIFORM_ETA_NBLOCKS, &state);

  ctr = rej_eta(a->coeffs, N, buf, buflen);

  while(ctr < N) {
    stream256_squeezeblocks(buf, 1, &state);
    ctr += rej_eta(a->coeffs + ctr, N - ctr, buf, STREAM256_BLOCKBYTES);
  }
}
void poly_uniform_eta_8x(poly *a0,
                         poly *a1,
                         poly *a2,
                         poly *a3,
                         poly *a4,
                         poly *a5,
                         poly *a6,
                         poly *a7,
                         const uint8_t seed[CRHBYTES],
                         uint16_t nonce0,
                         uint16_t nonce1,
                         uint16_t nonce2,
                         uint16_t nonce3,
                         uint16_t nonce4,
                         uint16_t nonce5,
                         uint16_t nonce6,
                         uint16_t nonce7)
{
  unsigned int ctr0, ctr1, ctr2, ctr3, ctr4, ctr5, ctr6, ctr7;
  ALIGNED_UINT8(POLY_UNIFORM_ETA_NBLOCKS*STREAM256_BLOCKBYTES) buf[8];

   __m512i f;

  f = _mm512_loadu_si512((__m512i *)seed);
  _mm512_store_si512(&buf[0].vec,f);
  _mm512_store_si512(&buf[1].vec,f);
  _mm512_store_si512(&buf[2].vec,f);
  _mm512_store_si512(&buf[3].vec,f);
  _mm512_store_si512(&buf[4].vec,f);
  _mm512_store_si512(&buf[5].vec,f);
  _mm512_store_si512(&buf[6].vec,f);
  _mm512_store_si512(&buf[7].vec,f);
 

  
  buf[0].coeffs[CRHBYTES + 0] = nonce0;
  buf[0].coeffs[CRHBYTES + 1] = nonce0 >> 8;
  buf[1].coeffs[CRHBYTES + 0] = nonce1;
  buf[1].coeffs[CRHBYTES + 1] = nonce1 >> 8;
  buf[2].coeffs[CRHBYTES + 0] = nonce2;
  buf[2].coeffs[CRHBYTES + 1] = nonce2 >> 8;
  buf[3].coeffs[CRHBYTES + 0] = nonce3;
  buf[3].coeffs[CRHBYTES + 1] = nonce3 >> 8;
  buf[4].coeffs[CRHBYTES + 0] = nonce4;
  buf[4].coeffs[CRHBYTES + 1] = nonce4 >> 8;
  buf[5].coeffs[CRHBYTES + 0] = nonce5;
  buf[5].coeffs[CRHBYTES + 1] = nonce5 >> 8;
  buf[6].coeffs[CRHBYTES + 0] = nonce6;
  buf[6].coeffs[CRHBYTES + 1] = nonce6 >> 8;
  buf[7].coeffs[CRHBYTES + 0] = nonce7;
  buf[7].coeffs[CRHBYTES + 1] = nonce7 >> 8;



  __m512i state[25];

  // shake256x8(state, buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs, buf[4].coeffs, buf[5].coeffs, buf[6].coeffs, buf[7].coeffs, REJ_UNIFORM_ETA_BUFLEN, buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs, buf[4].coeffs, buf[5].coeffs, buf[6].coeffs, buf[7].coeffs, 34); 
  shake256x8(state, buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs, buf[4].coeffs, buf[5].coeffs, buf[6].coeffs, buf[7].coeffs, POLY_UNIFORM_ETA_NBLOCKS*STREAM256_BLOCKBYTES, buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs, buf[4].coeffs, buf[5].coeffs, buf[6].coeffs, buf[7].coeffs, CRHBYTES+2); 

  ctr0 = rej_eta_avx(a0->coeffs, buf[0].coeffs);
  ctr1 = rej_eta_avx(a1->coeffs, buf[1].coeffs);
  ctr2 = rej_eta_avx(a2->coeffs, buf[2].coeffs);
  ctr3 = rej_eta_avx(a3->coeffs, buf[3].coeffs);
  ctr4 = rej_eta_avx(a4->coeffs, buf[4].coeffs);
  ctr5 = rej_eta_avx(a5->coeffs, buf[5].coeffs);
  ctr6 = rej_eta_avx(a6->coeffs, buf[6].coeffs);
  ctr7 = rej_eta_avx(a7->coeffs, buf[7].coeffs);
 
  while(ctr0 < N || ctr1 < N || ctr2 < N || ctr3 < N || ctr4 < N || ctr5 < N || ctr6 < N || ctr7 < N) {
    keccak_squeezeblocks8x(buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs, buf[4].coeffs, buf[5].coeffs, buf[6].coeffs, buf[7].coeffs, 1, state, SHAKE128_RATE);
    // keccak_squeezeblocks8x(buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs, buf[4].coeffs, buf[5].coeffs, buf[6].coeffs, buf[7].coeffs, 1, state, SHAKE256_RATE);
    ctr0 += rej_eta(a0->coeffs + ctr0, N - ctr0, buf[0].coeffs, SHAKE256_RATE);
    ctr1 += rej_eta(a1->coeffs + ctr1, N - ctr1, buf[1].coeffs, SHAKE256_RATE);
    ctr2 += rej_eta(a2->coeffs + ctr2, N - ctr2, buf[2].coeffs, SHAKE256_RATE);
    ctr3 += rej_eta(a3->coeffs + ctr3, N - ctr3, buf[3].coeffs, SHAKE256_RATE);
    ctr4 += rej_eta(a4->coeffs + ctr4, N - ctr4, buf[4].coeffs, SHAKE256_RATE);
    ctr5 += rej_eta(a5->coeffs + ctr5, N - ctr5, buf[5].coeffs, SHAKE256_RATE);
    ctr6 += rej_eta(a6->coeffs + ctr6, N - ctr6, buf[6].coeffs, SHAKE256_RATE);
    ctr7 += rej_eta(a7->coeffs + ctr7, N - ctr7, buf[7].coeffs, SHAKE256_RATE);
  }
}
/*************************************************
* Name:        poly_uniform_gamma1m1
*
* Description: Sample polynomial with uniformly random coefficients
*              in [-(GAMMA1 - 1), GAMMA1] by unpacking output stream
*              of SHAKE256(seed|nonce)
*
* Arguments:   - poly *a: pointer to output polynomial
*              - const uint8_t seed[]: byte array with seed of length CRHBYTES
*              - uint16_t nonce: 16-bit nonce
**************************************************/
#define POLY_UNIFORM_GAMMA1_NBLOCKS ((POLYZ_PACKEDBYTES + STREAM256_BLOCKBYTES - 1)/STREAM256_BLOCKBYTES)
void poly_uniform_gamma1(poly *a,
                         const uint8_t seed[CRHBYTES],
                         uint16_t nonce)
{
  uint8_t buf[POLY_UNIFORM_GAMMA1_NBLOCKS*STREAM256_BLOCKBYTES];
  stream256_state state;

  stream256_init(&state, seed, nonce);
  stream256_squeezeblocks(buf, POLY_UNIFORM_GAMMA1_NBLOCKS, &state);
  polyz_unpack(a, buf);
}
void poly_uniform_gamma1_8x(poly *a0,
                         poly *a1,
                         poly *a2,
                         poly *a3,
                         poly *a4,
                         poly *a5,
                         poly *a6,
                         poly *a7,
                         const uint8_t seed[CRHBYTES],
                         uint16_t nonce0,
                         uint16_t nonce1,
                         uint16_t nonce2,
                         uint16_t nonce3,
                         uint16_t nonce4,
                         uint16_t nonce5,
                         uint16_t nonce6,
                         uint16_t nonce7)
{
  ALIGNED_UINT8(POLY_UNIFORM_GAMMA1_NBLOCKS*STREAM256_BLOCKBYTES+14) buf[8];


  __m512i f;

  f = _mm512_loadu_si512((__m512i *)seed);
  _mm512_store_si512(&buf[0].vec,f);
  _mm512_store_si512(&buf[1].vec,f);
  _mm512_store_si512(&buf[2].vec,f);
  _mm512_store_si512(&buf[3].vec,f);
  _mm512_store_si512(&buf[4].vec,f);
  _mm512_store_si512(&buf[5].vec,f);
  _mm512_store_si512(&buf[6].vec,f);
  _mm512_store_si512(&buf[7].vec,f);
 

  buf[0].coeffs[CRHBYTES + 0] = nonce0;
  buf[0].coeffs[CRHBYTES + 1] = nonce0 >> 8;
  buf[1].coeffs[CRHBYTES + 0] = nonce1;
  buf[1].coeffs[CRHBYTES + 1] = nonce1 >> 8;
  buf[2].coeffs[CRHBYTES + 0] = nonce2;
  buf[2].coeffs[CRHBYTES + 1] = nonce2 >> 8;
  buf[3].coeffs[CRHBYTES + 0] = nonce3;
  buf[3].coeffs[CRHBYTES + 1] = nonce3 >> 8;
  buf[4].coeffs[CRHBYTES + 0] = nonce4;
  buf[4].coeffs[CRHBYTES + 1] = nonce4 >> 8;
  buf[5].coeffs[CRHBYTES + 0] = nonce5;
  buf[5].coeffs[CRHBYTES + 1] = nonce5 >> 8;
  buf[6].coeffs[CRHBYTES + 0] = nonce6;
  buf[6].coeffs[CRHBYTES + 1] = nonce6 >> 8;
  buf[7].coeffs[CRHBYTES + 0] = nonce7;
  buf[7].coeffs[CRHBYTES + 1] = nonce7 >> 8;

  __m512i state[25];
  shake256x8(state, buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs, buf[4].coeffs, buf[5].coeffs, buf[6].coeffs, buf[7].coeffs, POLY_UNIFORM_GAMMA1_NBLOCKS*STREAM256_BLOCKBYTES, buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs, buf[4].coeffs, buf[5].coeffs, buf[6].coeffs, buf[7].coeffs, CRHBYTES+2); 
  polyz_unpack(a0, buf[0].coeffs);
  polyz_unpack(a1, buf[1].coeffs);
  polyz_unpack(a2, buf[2].coeffs);
  polyz_unpack(a3, buf[3].coeffs);
  polyz_unpack(a4, buf[4].coeffs);
  polyz_unpack(a5, buf[5].coeffs);
  polyz_unpack(a6, buf[6].coeffs);
  polyz_unpack(a7, buf[7].coeffs);
  
}
/*************************************************
* Name:        challenge
*
* Description: Implementation of H. Samples polynomial with TAU nonzero
*              coefficients in {-1,1} using the output stream of
*              SHAKE256(seed).
*
* Arguments:   - poly *c: pointer to output polynomial
*              - const uint8_t mu[]: byte array containing seed of length SEEDBYTES
**************************************************/
void poly_challenge(poly *c, const uint8_t seed[SEEDBYTES]) {
  unsigned int i, b, pos;
  uint64_t signs;
  uint8_t buf[SHAKE256_RATE];
  keccak_state state;

  shake256_init(&state);
  shake256_absorb(&state, seed, SEEDBYTES);
  shake256_finalize(&state);
  shake256_squeezeblocks(buf, 1, &state);

  signs = 0;
  for(i = 0; i < 8; ++i)
    signs |= (uint64_t)buf[i] << 8*i;
  pos = 8;

  for(i = 0; i < N; ++i)
    c->coeffs[i] = 0;
  for(i = N-TAU; i < N; ++i) {
    do {
      if(pos >= SHAKE256_RATE) {
        shake256_squeezeblocks(buf, 1, &state);
        pos = 0;
      }

      b = buf[pos++];
    } while(b > i);

    c->coeffs[i] = c->coeffs[b];
    c->coeffs[b] = 1 - 2*(signs & 1);
    signs >>= 1;
  }
}

/*************************************************
* Name:        polyeta_pack
*
* Description: Bit-pack polynomial with coefficients in [-ETA,ETA].
*
* Arguments:   - uint8_t *r: pointer to output byte array with at least
*                            POLYETA_PACKEDBYTES bytes
*              - const poly *a: pointer to input polynomial
**************************************************/
void polyeta_pack(uint8_t *r, const poly *a) {
  unsigned int i;
  uint8_t t[8];
  DBENCH_START();

#if ETA == 2
  for(i = 0; i < N/8; ++i) {
    t[0] = ETA - a->coeffs[8*i+0];
    t[1] = ETA - a->coeffs[8*i+1];
    t[2] = ETA - a->coeffs[8*i+2];
    t[3] = ETA - a->coeffs[8*i+3];
    t[4] = ETA - a->coeffs[8*i+4];
    t[5] = ETA - a->coeffs[8*i+5];
    t[6] = ETA - a->coeffs[8*i+6];
    t[7] = ETA - a->coeffs[8*i+7];

    r[3*i+0]  = (t[0] >> 0) | (t[1] << 3) | (t[2] << 6);
    r[3*i+1]  = (t[2] >> 2) | (t[3] << 1) | (t[4] << 4) | (t[5] << 7);
    r[3*i+2]  = (t[5] >> 1) | (t[6] << 2) | (t[7] << 5);
  }
#elif ETA == 4
  for(i = 0; i < N/2; ++i) {
    t[0] = ETA - a->coeffs[2*i+0];
    t[1] = ETA - a->coeffs[2*i+1];
    r[i] = t[0] | (t[1] << 4);
  }
#endif

  DBENCH_STOP(*tpack);
}

/*************************************************
* Name:        polyeta_unpack
*
* Description: Unpack polynomial with coefficients in [-ETA,ETA].
*
* Arguments:   - poly *r: pointer to output polynomial
*              - const uint8_t *a: byte array with bit-packed polynomial
**************************************************/
void polyeta_unpack(poly *r, const uint8_t *a) {
  unsigned int i;
  DBENCH_START();

#if ETA == 2
  for(i = 0; i < N/8; ++i) {
    r->coeffs[8*i+0] =  (a[3*i+0] >> 0) & 7;
    r->coeffs[8*i+1] =  (a[3*i+0] >> 3) & 7;
    r->coeffs[8*i+2] = ((a[3*i+0] >> 6) | (a[3*i+1] << 2)) & 7;
    r->coeffs[8*i+3] =  (a[3*i+1] >> 1) & 7;
    r->coeffs[8*i+4] =  (a[3*i+1] >> 4) & 7;
    r->coeffs[8*i+5] = ((a[3*i+1] >> 7) | (a[3*i+2] << 1)) & 7;
    r->coeffs[8*i+6] =  (a[3*i+2] >> 2) & 7;
    r->coeffs[8*i+7] =  (a[3*i+2] >> 5) & 7;

    r->coeffs[8*i+0] = ETA - r->coeffs[8*i+0];
    r->coeffs[8*i+1] = ETA - r->coeffs[8*i+1];
    r->coeffs[8*i+2] = ETA - r->coeffs[8*i+2];
    r->coeffs[8*i+3] = ETA - r->coeffs[8*i+3];
    r->coeffs[8*i+4] = ETA - r->coeffs[8*i+4];
    r->coeffs[8*i+5] = ETA - r->coeffs[8*i+5];
    r->coeffs[8*i+6] = ETA - r->coeffs[8*i+6];
    r->coeffs[8*i+7] = ETA - r->coeffs[8*i+7];
  }
#elif ETA == 4
  for(i = 0; i < N/2; ++i) {
    r->coeffs[2*i+0] = a[i] & 0x0F;
    r->coeffs[2*i+1] = a[i] >> 4;
    r->coeffs[2*i+0] = ETA - r->coeffs[2*i+0];
    r->coeffs[2*i+1] = ETA - r->coeffs[2*i+1];
  }
#endif

  DBENCH_STOP(*tpack);
}

/*************************************************
* Name:        polyt1_pack
*
* Description: Bit-pack polynomial t1 with coefficients fitting in 10 bits.
*              Input coefficients are assumed to be standard representatives.
*
* Arguments:   - uint8_t *r: pointer to output byte array with at least
*                            POLYT1_PACKEDBYTES bytes
*              - const poly *a: pointer to input polynomial
**************************************************/
void polyt1_pack(uint8_t *r, const poly *a) {
  unsigned int i;
  DBENCH_START();

  for(i = 0; i < N/4; ++i) {
    r[5*i+0] = (a->coeffs[4*i+0] >> 0);
    r[5*i+1] = (a->coeffs[4*i+0] >> 8) | (a->coeffs[4*i+1] << 2);
    r[5*i+2] = (a->coeffs[4*i+1] >> 6) | (a->coeffs[4*i+2] << 4);
    r[5*i+3] = (a->coeffs[4*i+2] >> 4) | (a->coeffs[4*i+3] << 6);
    r[5*i+4] = (a->coeffs[4*i+3] >> 2);
  }

  DBENCH_STOP(*tpack);
}

/*************************************************
* Name:        polyt1_unpack
*
* Description: Unpack polynomial t1 with 10-bit coefficients.
*              Output coefficients are standard representatives.
*
* Arguments:   - poly *r: pointer to output polynomial
*              - const uint8_t *a: byte array with bit-packed polynomial
**************************************************/
void polyt1_unpack(poly *r, const uint8_t *a) {
  unsigned int i;
  DBENCH_START();

  for(i = 0; i < N/4; ++i) {
    r->coeffs[4*i+0] = ((a[5*i+0] >> 0) | ((uint32_t)a[5*i+1] << 8)) & 0x3FF;
    r->coeffs[4*i+1] = ((a[5*i+1] >> 2) | ((uint32_t)a[5*i+2] << 6)) & 0x3FF;
    r->coeffs[4*i+2] = ((a[5*i+2] >> 4) | ((uint32_t)a[5*i+3] << 4)) & 0x3FF;
    r->coeffs[4*i+3] = ((a[5*i+3] >> 6) | ((uint32_t)a[5*i+4] << 2)) & 0x3FF;
  }

  DBENCH_STOP(*tpack);
}

/*************************************************
* Name:        polyt0_pack
*
* Description: Bit-pack polynomial t0 with coefficients in ]-2^{D-1}, 2^{D-1}].
*
* Arguments:   - uint8_t *r: pointer to output byte array with at least
*                            POLYT0_PACKEDBYTES bytes
*              - const poly *a: pointer to input polynomial
**************************************************/
void polyt0_pack(uint8_t *r, const poly *a) {
  unsigned int i;
  uint32_t t[8];
  DBENCH_START();

  for(i = 0; i < N/8; ++i) {
    t[0] = (1 << (D-1)) - a->coeffs[8*i+0];
    t[1] = (1 << (D-1)) - a->coeffs[8*i+1];
    t[2] = (1 << (D-1)) - a->coeffs[8*i+2];
    t[3] = (1 << (D-1)) - a->coeffs[8*i+3];
    t[4] = (1 << (D-1)) - a->coeffs[8*i+4];
    t[5] = (1 << (D-1)) - a->coeffs[8*i+5];
    t[6] = (1 << (D-1)) - a->coeffs[8*i+6];
    t[7] = (1 << (D-1)) - a->coeffs[8*i+7];

    r[13*i+ 0]  =  t[0];
    r[13*i+ 1]  =  t[0] >>  8;
    r[13*i+ 1] |=  t[1] <<  5;
    r[13*i+ 2]  =  t[1] >>  3;
    r[13*i+ 3]  =  t[1] >> 11;
    r[13*i+ 3] |=  t[2] <<  2;
    r[13*i+ 4]  =  t[2] >>  6;
    r[13*i+ 4] |=  t[3] <<  7;
    r[13*i+ 5]  =  t[3] >>  1;
    r[13*i+ 6]  =  t[3] >>  9;
    r[13*i+ 6] |=  t[4] <<  4;
    r[13*i+ 7]  =  t[4] >>  4;
    r[13*i+ 8]  =  t[4] >> 12;
    r[13*i+ 8] |=  t[5] <<  1;
    r[13*i+ 9]  =  t[5] >>  7;
    r[13*i+ 9] |=  t[6] <<  6;
    r[13*i+10]  =  t[6] >>  2;
    r[13*i+11]  =  t[6] >> 10;
    r[13*i+11] |=  t[7] <<  3;
    r[13*i+12]  =  t[7] >>  5;
  }

  DBENCH_STOP(*tpack);
}

/*************************************************
* Name:        polyt0_unpack
*
* Description: Unpack polynomial t0 with coefficients in ]-2^{D-1}, 2^{D-1}].
*
* Arguments:   - poly *r: pointer to output polynomial
*              - const uint8_t *a: byte array with bit-packed polynomial
**************************************************/
void polyt0_unpack(poly *r, const uint8_t *a) {
  unsigned int i;
  DBENCH_START();

  for(i = 0; i < N/8; ++i) {
    r->coeffs[8*i+0]  = a[13*i+0];
    r->coeffs[8*i+0] |= (uint32_t)a[13*i+1] << 8;
    r->coeffs[8*i+0] &= 0x1FFF;

    r->coeffs[8*i+1]  = a[13*i+1] >> 5;
    r->coeffs[8*i+1] |= (uint32_t)a[13*i+2] << 3;
    r->coeffs[8*i+1] |= (uint32_t)a[13*i+3] << 11;
    r->coeffs[8*i+1] &= 0x1FFF;

    r->coeffs[8*i+2]  = a[13*i+3] >> 2;
    r->coeffs[8*i+2] |= (uint32_t)a[13*i+4] << 6;
    r->coeffs[8*i+2] &= 0x1FFF;

    r->coeffs[8*i+3]  = a[13*i+4] >> 7;
    r->coeffs[8*i+3] |= (uint32_t)a[13*i+5] << 1;
    r->coeffs[8*i+3] |= (uint32_t)a[13*i+6] << 9;
    r->coeffs[8*i+3] &= 0x1FFF;

    r->coeffs[8*i+4]  = a[13*i+6] >> 4;
    r->coeffs[8*i+4] |= (uint32_t)a[13*i+7] << 4;
    r->coeffs[8*i+4] |= (uint32_t)a[13*i+8] << 12;
    r->coeffs[8*i+4] &= 0x1FFF;

    r->coeffs[8*i+5]  = a[13*i+8] >> 1;
    r->coeffs[8*i+5] |= (uint32_t)a[13*i+9] << 7;
    r->coeffs[8*i+5] &= 0x1FFF;

    r->coeffs[8*i+6]  = a[13*i+9] >> 6;
    r->coeffs[8*i+6] |= (uint32_t)a[13*i+10] << 2;
    r->coeffs[8*i+6] |= (uint32_t)a[13*i+11] << 10;
    r->coeffs[8*i+6] &= 0x1FFF;

    r->coeffs[8*i+7]  = a[13*i+11] >> 3;
    r->coeffs[8*i+7] |= (uint32_t)a[13*i+12] << 5;
    r->coeffs[8*i+7] &= 0x1FFF;

    r->coeffs[8*i+0] = (1 << (D-1)) - r->coeffs[8*i+0];
    r->coeffs[8*i+1] = (1 << (D-1)) - r->coeffs[8*i+1];
    r->coeffs[8*i+2] = (1 << (D-1)) - r->coeffs[8*i+2];
    r->coeffs[8*i+3] = (1 << (D-1)) - r->coeffs[8*i+3];
    r->coeffs[8*i+4] = (1 << (D-1)) - r->coeffs[8*i+4];
    r->coeffs[8*i+5] = (1 << (D-1)) - r->coeffs[8*i+5];
    r->coeffs[8*i+6] = (1 << (D-1)) - r->coeffs[8*i+6];
    r->coeffs[8*i+7] = (1 << (D-1)) - r->coeffs[8*i+7];
  }

  DBENCH_STOP(*tpack);
}

/*************************************************
* Name:        polyz_pack
*
* Description: Bit-pack polynomial with coefficients
*              in [-(GAMMA1 - 1), GAMMA1].
*
* Arguments:   - uint8_t *r: pointer to output byte array with at least
*                            POLYZ_PACKEDBYTES bytes
*              - const poly *a: pointer to input polynomial
**************************************************/
void polyz_pack(uint8_t *r, const poly *a) {
  unsigned int i;
  uint32_t t[4];
  DBENCH_START();

#if GAMMA1 == (1 << 17)
  for(i = 0; i < N/4; ++i) {
    t[0] = GAMMA1 - a->coeffs[4*i+0];
    t[1] = GAMMA1 - a->coeffs[4*i+1];
    t[2] = GAMMA1 - a->coeffs[4*i+2];
    t[3] = GAMMA1 - a->coeffs[4*i+3];

    r[9*i+0]  = t[0];
    r[9*i+1]  = t[0] >> 8;
    r[9*i+2]  = t[0] >> 16;
    r[9*i+2] |= t[1] << 2;
    r[9*i+3]  = t[1] >> 6;
    r[9*i+4]  = t[1] >> 14;
    r[9*i+4] |= t[2] << 4;
    r[9*i+5]  = t[2] >> 4;
    r[9*i+6]  = t[2] >> 12;
    r[9*i+6] |= t[3] << 6;
    r[9*i+7]  = t[3] >> 2;
    r[9*i+8]  = t[3] >> 10;
  }
#elif GAMMA1 == (1 << 19)
  for(i = 0; i < N/2; ++i) {
    t[0] = GAMMA1 - a->coeffs[2*i+0];
    t[1] = GAMMA1 - a->coeffs[2*i+1];

    r[5*i+0]  = t[0];
    r[5*i+1]  = t[0] >> 8;
    r[5*i+2]  = t[0] >> 16;
    r[5*i+2] |= t[1] << 4;
    r[5*i+3]  = t[1] >> 4;
    r[5*i+4]  = t[1] >> 12;
  }
#endif

  DBENCH_STOP(*tpack);
}

/*************************************************
* Name:        polyz_unpack
*
* Description: Unpack polynomial z with coefficients
*              in [-(GAMMA1 - 1), GAMMA1].
*
* Arguments:   - poly *r: pointer to output polynomial
*              - const uint8_t *a: byte array with bit-packed polynomial
**************************************************/
// void polyz_unpack(poly *r, const uint8_t *a) {
//   unsigned int i;
//   DBENCH_START();

// #if GAMMA1 == (1 << 17)
//   for(i = 0; i < N/4; ++i) {
//     r->coeffs[4*i+0]  = a[9*i+0];
//     r->coeffs[4*i+0] |= (uint32_t)a[9*i+1] << 8;
//     r->coeffs[4*i+0] |= (uint32_t)a[9*i+2] << 16;
//     r->coeffs[4*i+0] &= 0x3FFFF;

//     r->coeffs[4*i+1]  = a[9*i+2] >> 2;
//     r->coeffs[4*i+1] |= (uint32_t)a[9*i+3] << 6;
//     r->coeffs[4*i+1] |= (uint32_t)a[9*i+4] << 14;
//     r->coeffs[4*i+1] &= 0x3FFFF;

//     r->coeffs[4*i+2]  = a[9*i+4] >> 4;
//     r->coeffs[4*i+2] |= (uint32_t)a[9*i+5] << 4;
//     r->coeffs[4*i+2] |= (uint32_t)a[9*i+6] << 12;
//     r->coeffs[4*i+2] &= 0x3FFFF;

//     r->coeffs[4*i+3]  = a[9*i+6] >> 6;
//     r->coeffs[4*i+3] |= (uint32_t)a[9*i+7] << 2;
//     r->coeffs[4*i+3] |= (uint32_t)a[9*i+8] << 10;
//     r->coeffs[4*i+3] &= 0x3FFFF;

//     r->coeffs[4*i+0] = GAMMA1 - r->coeffs[4*i+0];
//     r->coeffs[4*i+1] = GAMMA1 - r->coeffs[4*i+1];
//     r->coeffs[4*i+2] = GAMMA1 - r->coeffs[4*i+2];
//     r->coeffs[4*i+3] = GAMMA1 - r->coeffs[4*i+3];
//   }
// #elif GAMMA1 == (1 << 19)
//   for(i = 0; i < N/2; ++i) {
//     r->coeffs[2*i+0]  = a[5*i+0];
//     r->coeffs[2*i+0] |= (uint32_t)a[5*i+1] << 8;
//     r->coeffs[2*i+0] |= (uint32_t)a[5*i+2] << 16;
//     r->coeffs[2*i+0] &= 0xFFFFF;

//     r->coeffs[2*i+1]  = a[5*i+2] >> 4;
//     r->coeffs[2*i+1] |= (uint32_t)a[5*i+3] << 4;
//     r->coeffs[2*i+1] |= (uint32_t)a[5*i+4] << 12;
//     r->coeffs[2*i+0] &= 0xFFFFF;

//     r->coeffs[2*i+0] = GAMMA1 - r->coeffs[2*i+0];
//     r->coeffs[2*i+1] = GAMMA1 - r->coeffs[2*i+1];
//   }
// #endif

//   DBENCH_STOP(*tpack);
// }
#if GAMMA1 == (1 << 17)
 void polyz_unpack(poly * restrict r, const uint8_t *a) {
  unsigned int i;
  __m512i f;
  const __m512i zero = _mm512_setzero_si512();
  const __m512i mask  = _mm512_set1_epi32(0x3FFFF);
  const __m512i gamma1 = _mm512_set1_epi32(GAMMA1);
  const __m512i idx8  = _mm512_set_epi8(48,35,34,33,48,33,32,31,
                                        48,31,30,29,48,29,28,27,
                                        48,26,25,24,48,24,23,22,
                                        48,22,21,20,48,20,19,18,
                                        48,17,16,15,48,15,14,13,
                                        48,13,12,11,48,11,10, 9,
                                        48, 8, 7, 6,48, 6, 5, 4,
                                        48, 4, 3, 2,48, 2, 1, 0);
   const __m512i srlvdidx = _mm512_set_epi32(6,4,2,0,6,4,2,0,6,4,2,0,6,4,2,0);                                      
  DBENCH_START();

  for(i = 0; i < N/16; i++) {
    f = _mm512_loadu_si512((__m512i *)&a[36*i]);
    f = _mm512_mask_blend_epi64(0x1F, zero, f);
    f = _mm512_permutexvar_epi8(idx8, f);
    f = _mm512_srlv_epi32(f,srlvdidx);
    f = _mm512_and_si512(f,mask);
    f = _mm512_sub_epi32(gamma1,f);
    _mm512_store_si512(&r->vec[i],f);
  }

  DBENCH_STOP(*tpack);
}
#elif GAMMA1 == (1 << 19)
void polyz_unpack(poly * restrict r, const uint8_t *a) {
  unsigned int i;
  __m512i f;
  const __m512i zero = _mm512_setzero_si512();
  const __m512i mask  = _mm512_set1_epi32(0xFFFFF);
  const __m512i gamma1 = _mm512_set1_epi32(GAMMA1);
  const __m512i idx8  = _mm512_set_epi8(48,39,38,37,48,37,36,35,
                                        48,34,33,32,48,32,31,30,
                                        48,29,28,27,48,27,26,25,
                                        48,24,23,22,48,22,21,20,
                                        48,19,18,17,48,17,16,15,
                                        48,14,13,12,48,12,11,10,
                                        48, 9, 8, 7,48, 7, 6, 5,
                                        48, 4, 3, 2,48, 2, 1, 0);
   const __m512i srlvdidx = _mm512_set_epi32(4,0,4,0,4,0,4,0,4,0,4,0,4,0,4,0);                                      
  DBENCH_START();

  for(i = 0; i < N/16; i++) {
    f = _mm512_loadu_si512((__m512i *)&a[40*i]);
    f = _mm512_mask_blend_epi64(0x1F, zero, f);
    f = _mm512_permutexvar_epi8(idx8, f);
    f = _mm512_srlv_epi32(f,srlvdidx);
    f = _mm512_and_si512(f,mask);
    f = _mm512_sub_epi32(gamma1,f);
    _mm512_store_si512(&r->vec[i],f);
  }

  DBENCH_STOP(*tpack);
}
 #endif
/*************************************************
* Name:        polyw1_pack
*
* Description: Bit-pack polynomial w1 with coefficients in [0,15] or [0,43].
*              Input coefficients are assumed to be standard representatives.
*
* Arguments:   - uint8_t *r: pointer to output byte array with at least
*                            POLYW1_PACKEDBYTES bytes
*              - const poly *a: pointer to input polynomial
**************************************************/
// void polyw1_pack(uint8_t *r, const poly *a) {
//   unsigned int i;
//   DBENCH_START();

// #if GAMMA2 == (Q-1)/88
//   for(i = 0; i < N/4; ++i) {
//     r[3*i+0]  = a->coeffs[4*i+0];
//     r[3*i+0] |= a->coeffs[4*i+1] << 6;
//     r[3*i+1]  = a->coeffs[4*i+1] >> 2;
//     r[3*i+1] |= a->coeffs[4*i+2] << 4;
//     r[3*i+2]  = a->coeffs[4*i+2] >> 4;
//     r[3*i+2] |= a->coeffs[4*i+3] << 2;
//   }
// #elif GAMMA2 == (Q-1)/32
//   for(i = 0; i < N/2; ++i)
//     r[i] = a->coeffs[2*i+0] | (a->coeffs[2*i+1] << 4);
// #endif

//   DBENCH_STOP(*tpack);
// }
#if GAMMA2 == (Q-1)/88 //D2
void polyw1_pack(uint8_t *r, const poly * restrict a) {
  unsigned int i;
  __m512i f0,f1,f2,f3;
  const __m512i shift1 = _mm512_set1_epi16((64 << 8) + 1);
  const __m512i shift2 = _mm512_set1_epi32((4096 << 16) + 1);
  const __m512i shufdidx1 = _mm512_set_epi32(15,11,7,3,14,10,6,2,13,9,5,1,12,8,4,0);
  const __m512i shufdidx2 = _mm512_set_epi32(-1,-1,-1,-1,14,13,12,10,9,8,6,5,4,2,1,0);
  const __m512i shufbidx = _mm512_set_epi8(-1,-1,-1,-1,14,13,12,10, 9, 8, 6, 5, 4, 2, 1, 0,-1,-1,-1,-1,14,13,12,10, 9, 8, 6, 5, 4, 2, 1, 0,
                                           -1,-1,-1,-1,14,13,12,10, 9, 8, 6, 5, 4, 2, 1, 0,-1,-1,-1,-1,14,13,12,10, 9, 8, 6, 5, 4, 2, 1, 0);
  DBENCH_START();

  for(i = 0; i < N/64; i++) {
    //load coeffs 换到round3 应是a->coeffs[64*i+0] a->coeffs[64*i+16] a->coeffs[64*i+32] a->coeffs[64*i+48]
    f0 = _mm512_load_si512((__m512i *)&a->vec[4*i+0]);
    f1 = _mm512_load_si512((__m512i *)&a->vec[4*i+1]);
    f2 = _mm512_load_si512((__m512i *)&a->vec[4*i+2]);
    f3 = _mm512_load_si512((__m512i *)&a->vec[4*i+3]);
    f0 = _mm512_packus_epi32(f0,f1);
    f1 = _mm512_packus_epi32(f2,f3);
    f0 = _mm512_packus_epi16(f0,f1);
    f0 = _mm512_maddubs_epi16(f0,shift1);
    f0 = _mm512_madd_epi16(f0,shift2);
    f0 = _mm512_permutexvar_epi32(shufdidx1,f0);
    f0 = _mm512_shuffle_epi8(f0,shufbidx);
    f0 = _mm512_permutexvar_epi32(shufdidx2,f0);
    _mm512_storeu_si512((__m512i *)&r[48*i],f0);
  }
}
//D3/D5  w1 with coefficients in [0,15] NIST KAT test right
#elif GAMMA2 == (Q-1)/32
void polyw1_pack(uint8_t * restrict r, const poly * restrict a) {
  unsigned int i;
  __m512i f0, f1, f2, f3, f4, f5, f6, f7;
  const __m512i shift = _mm512_set1_epi16((16 << 8) + 1);
  const __m512i shufbidx1 = _mm512_set_epi32(15,11,7,3,14,10,6,2,13,9,5,1,12,8,4,0);
  const __m512i shufbidx = _mm512_set_epi8(15,14,11,10,7,6,3,2,13,12,9,8,5,4,1,0,
                                           15,14,11,10,7,6,3,2,13,12,9,8,5,4,1,0,
                                           15,14,11,10,7,6,3,2,13,12,9,8,5,4,1,0,
                                           15,14,11,10,7,6,3,2,13,12,9,8,5,4,1,0);
  DBENCH_START();

  for(i = 0; i < N/128; ++i) {
    //load coeffs 换到round3 应是a->coeffs[128*i+0] a->coeffs[128*i+16] a->coeffs[128*i+32] a->coeffs[128*i+48] a->coeffs[128*i+64] a->coeffs[128*i+80] a->coeffs[128*i+96] a->coeffs[128*i+112]
    f0 = _mm512_loadu_si512(&a->coeffs[128*i+0]);
    f1 = _mm512_loadu_si512(&a->coeffs[128*i+16]);
    f2 = _mm512_loadu_si512(&a->coeffs[128*i+32]);
    f3 = _mm512_loadu_si512(&a->coeffs[128*i+48]);
    f4 = _mm512_loadu_si512(&a->coeffs[128*i+64]);
    f5 = _mm512_loadu_si512(&a->coeffs[128*i+80]);
    f6 = _mm512_loadu_si512(&a->coeffs[128*i+96]);
    f7 = _mm512_loadu_si512(&a->coeffs[128*i+112]);
    f0 = _mm512_packus_epi32(f0,f1);
    f1 = _mm512_packus_epi32(f2,f3);
    f2 = _mm512_packus_epi32(f4,f5);
    f3 = _mm512_packus_epi32(f6,f7);
    f0 = _mm512_packus_epi16(f0,f1);
    f1 = _mm512_packus_epi16(f2,f3);
    f0 = _mm512_maddubs_epi16(f0,shift);
    f1 = _mm512_maddubs_epi16(f1,shift);
    f0 = _mm512_packus_epi16(f0,f1);
    f0 = _mm512_permutexvar_epi32(shufbidx1,f0);
    f0 = _mm512_shuffle_epi8(f0,shufbidx);
    _mm512_storeu_si512(&r[64*i], f0);
  }

  DBENCH_STOP(*tpack);
}
#endif