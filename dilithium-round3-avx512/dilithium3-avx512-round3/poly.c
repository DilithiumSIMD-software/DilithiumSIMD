#include <stdint.h>
#include "params.h"
#include "poly.h"
#include "ntt.h"
#include "reduce.h"
#include "rounding.h"
#include "symmetric.h"
#include "rejsample.h"
#include "fips202x8.h"
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
void poly_smallntt(poly *a) {
  DBENCH_START();

  smallntt_avx(a->vec, qdata.vec);

  DBENCH_STOP(*tmul);
}

void poly_tailoredntt(poly *a) {
  DBENCH_START();

  tailoredntt_avx(a->vec, qdata.vec);

  DBENCH_STOP(*tmul);
}

void poly_instailoredntt(poly *a) {
  DBENCH_START();

  instailoredntt_avx(a->vec, qdata.vec);

  DBENCH_STOP(*tmul);
}

/*************************************************
* Name:        poly_freeze
*
* Description: Inplace reduction of all coefficients of polynomial to
*              standard representatives.
*
* Arguments:   - poly *a: pointer to input/output polynomial
**************************************************/
void poly_freeze(poly *a) {
  unsigned int i;
  DBENCH_START();

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

  DBENCH_STOP(*tred);
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
void poly_ntt(poly *a) {
  DBENCH_START();

  ntt_avx(a->vec, qdata.vec);

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
void poly_invntt_tomont(poly *a) {
  DBENCH_START();

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
void poly_pointwise_montgomery(poly *c, const poly *a, const poly *b) {
  // unsigned int i;
  DBENCH_START();
  pointwise_avx(c->vec, a->vec, b->vec, qdata.vec);

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
void poly_power2round(poly *a1, poly *a0, const poly *a) {
  DBENCH_START();
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
void poly_decompose(poly *a1, poly *a0, const poly *a) {
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
unsigned int poly_make_hint(poly *h, const poly *a0, const poly *a1) {

   unsigned int r;
  r = make_hint_avx(h->coeffs, a0->coeffs, a1->coeffs);
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
void poly_use_hint(poly *b, const poly *a, const poly *h) {
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
*              output stream of SHAKE256(seed|nonce) or AES256CTR(seed,nonce).
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

    if(t0 < 9)
      a[ctr++] = 4 - t0;
    if(t1 < 9 && ctr < len)
      a[ctr++] = 4 - t1;
  }

  DBENCH_STOP(*tsample);
  return ctr;
}

/*************************************************
* Name:        poly_uniform_eta
*
* Description: Sample polynomial with uniformly random coefficients
*              in [-ETA,ETA] by performing rejection sampling on the
*              output stream from SHAKE256(seed|nonce) or AES256CTR(seed,nonce).
*
* Arguments:   - poly *a: pointer to output polynomial
*              - const uint8_t seed[]: byte array with seed of length SEEDBYTES
*              - uint16_t nonce: 2-byte nonce
**************************************************/

#define POLY_UNIFORM_ETA_NBLOCKS ((227 + STREAM128_BLOCKBYTES - 1)/STREAM128_BLOCKBYTES)
void poly_uniform_eta(poly *a,
                      const uint8_t seed[SEEDBYTES],
                      uint16_t nonce)
{
  unsigned int ctr;
  unsigned int buflen = POLY_UNIFORM_ETA_NBLOCKS*STREAM128_BLOCKBYTES;
  uint8_t buf[POLY_UNIFORM_ETA_NBLOCKS*STREAM128_BLOCKBYTES];
  stream128_state state;

  stream128_init(&state, seed, nonce);
  stream128_squeezeblocks(buf, POLY_UNIFORM_ETA_NBLOCKS, &state);

  ctr = rej_eta(a->coeffs, N, buf, buflen);

  while(ctr < N) {
    stream128_squeezeblocks(buf, 1, &state);
    ctr += rej_eta(a->coeffs + ctr, N - ctr, buf, STREAM128_BLOCKBYTES);
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
  ALIGNED_UINT8(REJ_UNIFORM_ETA_BUFLEN) buf[8];

   __m256i f;

  f = _mm256_loadu_si256((__m256i *)&seed[0]);
  _mm256_store_si256((__m256i *)&buf[0].coeffs[0],f);
  _mm256_store_si256((__m256i *)&buf[1].coeffs[0],f);
  _mm256_store_si256((__m256i *)&buf[2].coeffs[0],f);
  _mm256_store_si256((__m256i *)&buf[3].coeffs[0],f);
  _mm256_store_si256((__m256i *)&buf[4].coeffs[0],f);
  _mm256_store_si256((__m256i *)&buf[5].coeffs[0],f);
  _mm256_store_si256((__m256i *)&buf[6].coeffs[0],f);
  _mm256_store_si256((__m256i *)&buf[7].coeffs[0],f);


  
  buf[0].coeffs[SEEDBYTES+0] = nonce0;
  buf[0].coeffs[SEEDBYTES+1] = nonce0 >> 8;
  buf[1].coeffs[SEEDBYTES+0] = nonce1;
  buf[1].coeffs[SEEDBYTES+1] = nonce1 >> 8;
  buf[2].coeffs[SEEDBYTES+0] = nonce2;
  buf[2].coeffs[SEEDBYTES+1] = nonce2 >> 8;
  buf[3].coeffs[SEEDBYTES+0] = nonce3;
  buf[3].coeffs[SEEDBYTES+1] = nonce3 >> 8;
  buf[4].coeffs[SEEDBYTES+0] = nonce4;
  buf[4].coeffs[SEEDBYTES+1] = nonce4 >> 8;
  buf[5].coeffs[SEEDBYTES+0] = nonce5;
  buf[5].coeffs[SEEDBYTES+1] = nonce5 >> 8;
  buf[6].coeffs[SEEDBYTES+0] = nonce6;
  buf[6].coeffs[SEEDBYTES+1] = nonce6 >> 8;
  buf[7].coeffs[SEEDBYTES+0] = nonce7;
  buf[7].coeffs[SEEDBYTES+1] = nonce7 >> 8;


  __m512i state[25];

  shake128x8(state, buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs, buf[4].coeffs, buf[5].coeffs, buf[6].coeffs, buf[7].coeffs, REJ_UNIFORM_ETA_BUFLEN, buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs, buf[4].coeffs, buf[5].coeffs, buf[6].coeffs, buf[7].coeffs, 34); 

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
*              of SHAKE256(seed|nonce) or AES256CTR(seed,nonce).
*
* Arguments:   - poly *a: pointer to output polynomial
*              - const uint8_t seed[]: byte array with seed of length CRHBYTES
*              - uint16_t nonce: 16-bit nonce
**************************************************/

#define POLY_UNIFORM_GAMMA1_NBLOCKS ((640 + STREAM256_BLOCKBYTES - 1)/STREAM256_BLOCKBYTES)
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
                         const uint8_t seed[48],
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


  __m256i f;
  __m128i g;

  f = _mm256_load_si256((__m256i *)seed);
  _mm256_store_si256((__m256i *)buf[0].coeffs,f);
  _mm256_store_si256((__m256i *)buf[1].coeffs,f);
  _mm256_store_si256((__m256i *)buf[2].coeffs,f);
  _mm256_store_si256((__m256i *)buf[3].coeffs,f);
  _mm256_store_si256((__m256i *)buf[4].coeffs,f);
  _mm256_store_si256((__m256i *)buf[5].coeffs,f);
  _mm256_store_si256((__m256i *)buf[6].coeffs,f);
  _mm256_store_si256((__m256i *)buf[7].coeffs,f);
  g = _mm_load_si128((__m128i *)&seed[32]);
  _mm_store_si128((__m128i *)&buf[0].coeffs[32],g);
  _mm_store_si128((__m128i *)&buf[1].coeffs[32],g);
  _mm_store_si128((__m128i *)&buf[2].coeffs[32],g);
  _mm_store_si128((__m128i *)&buf[3].coeffs[32],g);
 _mm_store_si128((__m128i *)&buf[4].coeffs[32],g);
  _mm_store_si128((__m128i *)&buf[5].coeffs[32],g);
  _mm_store_si128((__m128i *)&buf[6].coeffs[32],g);
  _mm_store_si128((__m128i *)&buf[7].coeffs[32],g);

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
  int8_t t[N];
  for(i = 0; i < N; i ++)
  {
    t[i] = a->coeffs[i];
  }
  __m512i idx64 = _mm512_set_epi8(63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 62, 60, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
  __m512i s_vec, tmp;
  __m512i mask_vec;
  mask_vec = _mm512_set1_epi16(0xff);
  const __m512i eta = _mm512_set1_epi8(4);
  for(i = 0; i < 4; i ++)
  {
    s_vec = _mm512_loadu_si512(&t[i*64]);
    s_vec = _mm512_sub_epi8(eta, s_vec);
    tmp = _mm512_srli_epi16(s_vec, 4);
    s_vec = _mm512_xor_si512(s_vec, tmp);
    s_vec = _mm512_and_si512(s_vec, mask_vec);
    s_vec = _mm512_permutexvar_epi8(idx64, s_vec);
    _mm512_storeu_epi8(&r[32*i], s_vec);
  }
}

/*************************************************
* Name:        polyeta_unpack
*
* Description: Unpack polynomial with coefficients in [-ETA,ETA].
*
* Arguments:   - poly *r: pointer to output polynomial
*              - const uint8_t *a: byte array with bit-packed polynomial
**************************************************/

void polyeta_unpack(poly *res, const uint8_t *a)
{
  int8_t r[N];
  const __m512i idx64 = _mm512_set_epi8(63, 31, 63, 30, 63, 29, 63, 28, 63, 27, 63, 26, 63, 25, 63, 24, 63, 23, 63, 22, 63, 21, 63, 20, 63, 19, 63, 18, 63, 17, 63, 16, 63, 15, 63, 14, 63, 13, 63, 12, 63, 11, 63, 10, 63, 9, 63, 8, 63, 7, 63, 6, 63, 5, 63, 4, 63, 3, 63, 2, 63, 1, 63, 0);
  const __m512i eta_vec = _mm512_set1_epi8(4);
  __m512i mask_vec1 = _mm512_set1_epi16(0x0f00);
  __m512i mask_vec2 = _mm512_set1_epi16(0x000f);
  
  __mmask64 mask = 0x5555555555555555;
  __m512i f0, f1;
  for(int i = 0; i < 4; i ++)
  {
    f0 = _mm512_loadu_si512(&a[32*i]);
    f0 = _mm512_maskz_permutexvar_epi8(mask, idx64, f0);
    f1 = _mm512_slli_epi16(f0, 4); 
    f1 = _mm512_and_si512(f1, mask_vec1);
    f0 = _mm512_and_si512(f0, mask_vec2);
    f0 = _mm512_xor_si512(f0, f1);
    f0 = _mm512_sub_epi8(eta_vec, f0);
    _mm512_storeu_si512(&r[64*i], f0);
  }
  for(int i = 0; i < N; i ++)
  {
    res->coeffs[i] = r[i];
  }
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
  uint16_t a0[N];
  for(int i = 0; i < N; i ++)
  {
    a0[i] = a->coeffs[i];
  }
  __m512i f0, f1;
  __m512i mask_vec1 = _mm512_set1_epi32(0xffff0000);
  __m512i mask_vec2 = _mm512_set1_epi32(0x0000ffff);
  __m512i mask_vec3 = _mm512_set1_epi64(0xffffffff00000000);
  __m512i mask_vec4 = _mm512_set1_epi64(0x00000000ffffffff);
  
  const __m512i idx64  = _mm512_set_epi8(63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 60, 59, 58, 57, 56, 52, 51, 50, 49, 48, 44, 43, 42, 41, 40, 36, 35, 34, 33, 32, 28, 27, 26, 25, 24, 20, 19, 18, 17, 16, 12, 11, 10, 9, 8, 4, 3, 2, 1, 0);
  
  for(int i = 0; i < 8; i ++)
  {
    f0 = _mm512_load_si512(&a0[i*32]);
    f1 = _mm512_and_si512(f0, mask_vec1);
    f1 = _mm512_srli_epi32(f1, 6); 
    f0 = _mm512_and_si512(f0, mask_vec2);
    f0 = _mm512_xor_si512(f0, f1);
    f1 = _mm512_and_si512(f0, mask_vec3);
    f1 = _mm512_srli_epi64(f1, 12); 
    f0 = _mm512_and_si512(f0, mask_vec4);
    f0 = _mm512_xor_si512(f0, f1);
    f0 = _mm512_permutexvar_epi8(idx64, f0);
    _mm512_storeu_si512(&r[40*i], f0);
  }
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
  __m512i f0, f1;
  int16_t r1[N];
  const __m512i idx64  = _mm512_set_epi8(63, 63, 63, 39, 38, 37, 36, 35, 63, 63, 63, 34, 33, 32, 31, 30, 63, 63, 63, 29, 28, 27, 26, 25, 63, 63, 63, 24, 23, 22, 21, 20, 63, 63, 63, 19, 18, 17, 16, 15, 63, 63, 63, 14, 13, 12, 11, 10, 63, 63, 63, 9, 8, 7, 6, 5, 63, 63, 63, 4, 3, 2, 1, 0);
  __m512i mask_vec1 = _mm512_set1_epi32(0x03ff0000);
  __m512i mask_vec2 = _mm512_set1_epi32(0x000003ff);
  __m512i mask_vec3 = _mm512_set1_epi64(0x000fffff00000000);
  __m512i mask_vec4 = _mm512_set1_epi64(0x00000000000fffff);
  __mmask64 mask = 0x1f1f1f1f1f1f1f1f;
  for(int i = 0; i < 8; i ++)
  {
    f0 = _mm512_loadu_si512(&a[i*40]);
    f0 = _mm512_maskz_permutexvar_epi8(mask, idx64, f0);
   
    //split 20 bits
    f1 = _mm512_slli_epi64(f0, 12); 
    f1 = _mm512_and_si512(f1, mask_vec3);
    f0 = _mm512_and_si512(f0, mask_vec4);
    f0 = _mm512_xor_si512(f0, f1);
    // //split 10 bits
    f1 = _mm512_slli_epi32(f0, 6); 
    f1 = _mm512_and_si512(f1, mask_vec1);
    f0 = _mm512_and_si512(f0, mask_vec2);
    f0 = _mm512_xor_si512(f0, f1);
    _mm512_storeu_si512(&r1[32*i], f0);
  }
  for(int i = 0; i < N; i ++)
  {
    r->coeffs[i] = r1[i];
  }
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