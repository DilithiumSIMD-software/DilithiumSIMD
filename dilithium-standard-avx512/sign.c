#include <stdint.h>
#include "params.h"
#include "sign.h"
#include "packing.h"
#include "polyvec.h"
#include "poly.h"
#include "randombytes.h"
#include "symmetric.h"
#include "fips202.h"
#include "mult.h"
#include "SHA-3_256.h"
#include <string.h>
/*************************************************
* Name:        crypto_sign_keypair
*
* Description: Generates public and private key.
*
* Arguments:   - uint8_t *pk: pointer to output public key (allocated
*                             array of CRYPTO_PUBLICKEYBYTES bytes)
*              - uint8_t *sk: pointer to output private key (allocated
*                             array of CRYPTO_SECRETKEYBYTES bytes)
*
* Returns 0 (success)
**************************************************/
int crypto_sign_keypair(uint8_t *pk, uint8_t *sk) {
  uint8_t seedbuf[2*SEEDBYTES + CRHBYTES];
  uint8_t tr[TRBYTES];
  const uint8_t *rho, *rhoprime, *key;
  polyvecl mat[K];
  polyvecl s1;
  polyveck s2, t1, t0;
  int i;
  /* Get randomness for rho, rhoprime and key */
  randombytes(seedbuf, SEEDBYTES);
  shake256_avx512seq(seedbuf, 2*SEEDBYTES + CRHBYTES, seedbuf, SEEDBYTES);
  rho = seedbuf;
  rhoprime = rho + SEEDBYTES;
  key = rhoprime + CRHBYTES;

   /* Store rho, key */
  memcpy(pk, rho, SEEDBYTES);
  memcpy(sk, rho, SEEDBYTES);
  memcpy(sk + SEEDBYTES, key, SEEDBYTES);

  /* Expand matrix */
  polyvec_matrix_expand(mat, rho);

  /* Sample short vectors s1 and s2 */

  #if K == 4 && L == 4
    poly_uniform_eta_8x(&s1.vec[0], &s1.vec[1], &s1.vec[2], &s1.vec[3], &s2.vec[0], &s2.vec[1], &s2.vec[2], &s2.vec[3], rhoprime, 0, 1, 2, 3, 4, 5, 6, 7);
  #elif K == 6 && L == 5
    poly tmp;
    poly_uniform_eta_8x(&s1.vec[0], &s1.vec[1], &s1.vec[2], &s1.vec[3], &s1.vec[4], &s2.vec[0], &s2.vec[1], &s2.vec[2], rhoprime, 0, 1, 2, 3, 4, 5, 6, 7);
    poly_uniform_eta_8x(&s2.vec[3], &s2.vec[4], &s2.vec[5], &tmp, &tmp, &tmp, &tmp, &tmp, rhoprime, 8, 9, 10, 11, 12, 13, 14, 15);
  #elif K == 8 && L == 7
    poly tmp;
    poly_uniform_eta_8x(&s1.vec[0], &s1.vec[1], &s1.vec[2], &s1.vec[3], &s1.vec[4], &s1.vec[5], &s1.vec[6], &tmp, rhoprime, 0, 1, 2, 3, 4, 5, 6, 0);
    poly_uniform_eta_8x(&s2.vec[0], &s2.vec[1], &s2.vec[2], &s2.vec[3], &s2.vec[4], &s2.vec[5], &s2.vec[6], &s2.vec[7], rhoprime, 7, 8, 9, 10, 11, 12, 13, 14);
  #else
  #error
  #endif
  /* Pack short vectors s1 and s2 */
  for(i = 0; i < L; i++)
    polyeta_pack(sk + 2*SEEDBYTES + CRHBYTES + i*POLYETA_PACKEDBYTES, &s1.vec[i]);
  for(i = 0; i < K; i++)
    polyeta_pack(sk + 2*SEEDBYTES + CRHBYTES + (L + i)*POLYETA_PACKEDBYTES, &s2.vec[i]);

  /* Matrix-vector multiplication */
  polyvecl_smallntt(&s1);
  polyvec_matrix_pointwise_montgomery(&t1, mat, &s1);
  polyveck_invntt_tomont(&t1);

  /* Add error vector s2 */
  polyveck_add(&t1, &t1, &s2);

  /* Extract t1 and write public key */
  polyveck_caddq(&t1);
  polyveck_power2round(&t1, &t0, &t1);
  for(i = 0; i < K; i ++)
  {
    polyt1_pack(pk + SEEDBYTES + i*POLYT1_PACKEDBYTES, &t1.vec[i]);
    polyt0_pack(sk + 2*SEEDBYTES + CRHBYTES + (L+K)*POLYETA_PACKEDBYTES + i*POLYT0_PACKEDBYTES, &t0.vec[i]);
  }
  /* Compute H(rho, t1) and write secret key */
  shake256_avx512seq(tr, TRBYTES, pk, CRYPTO_PUBLICKEYBYTES);
  memcpy(sk+2*SEEDBYTES, tr, TRBYTES);
  return 0;
}

/*************************************************
* Name:        crypto_sign_signature
*
* Description: Computes signature.
*
* Arguments:   - uint8_t *sig:   pointer to output signature (of length CRYPTO_BYTES)
*              - size_t *siglen: pointer to output length of signature
*              - uint8_t *m:     pointer to message to be signed
*              - size_t mlen:    length of message
*              - uint8_t *sk:    pointer to bit-packed secret key
*
* Returns 0 (success)
**************************************************/
int crypto_sign_signature(uint8_t *sig,
                          size_t *siglen,
                          const uint8_t *m,
                          size_t mlen,
                          const uint8_t *sk)
{
  unsigned int n;
  __attribute__((aligned(32))) uint8_t seedbuf[2*SEEDBYTES + TRBYTES + RNDBYTES + 2*CRHBYTES];
  uint8_t *rho, *tr, *key, *mu, *rhoprime, *rnd;
  uint16_t nonce = 0;
  polyvecl mat[K], s1, y, z;
  polyveck t0, s2, w1, w0, h;
  poly cp, tmp;
  #if K == 4
  __attribute__((aligned(32))) uint32_t s1_table[2*N];
  __attribute__((aligned(32))) uint32_t s2_table[2*N];
  #elif K == 6
  __attribute__((aligned(32))) uint32_t s11_table[2*N], s12_table[2*N];
  __attribute__((aligned(32))) uint32_t s21_table[2*N], s22_table[2*N];
  #elif K == 8
  __attribute__((aligned(32))) uint32_t s11_table[2*N];
  __attribute__((aligned(32))) uint32_t s12_table[2*N];
  __attribute__((aligned(32))) uint32_t s21_table[2*N];
  __attribute__((aligned(32))) uint32_t s22_table[2*N];
  #endif
  rho = seedbuf;
  tr = rho + SEEDBYTES;
  key = tr + TRBYTES;
  rnd = key + SEEDBYTES;
  mu = rnd + RNDBYTES;
  rhoprime = mu + CRHBYTES;
  unpack_sk(rho, tr, key, &t0, &s1, &s2, sk);

  #if DILITHIUM_MODE == 2
  prepare_s1_table(s1_table, &s1);
  prepare_s2_table(s2_table, &s2);
  #elif DILITHIUM_MODE == 3
  prepare_s1_table_32_avx512(s11_table, s12_table, &s1);
  prepare_s2_table_32_avx512(s21_table, s22_table, &s2);
  #elif DILITHIUM_MODE == 5
  prepare_s_table4x(s11_table, &s1.vec[0]);
  prepare_s_table3x(s12_table, &s1.vec[4]);
  prepare_s_table4x(s21_table, &s2.vec[0]);
  prepare_s_table4x(s22_table, &s2.vec[4]);
  #endif


  /* Compute mu = CRH(tr, msg) */ 
  uint8_t input[TRBYTES+mlen];
  memcpy(input, tr, TRBYTES);
  memcpy(input+TRBYTES, m, mlen);
  shake256_avx512seq(mu, CRHBYTES, input, TRBYTES+mlen);

#ifdef DILITHIUM_RANDOMIZED_SIGNING
  randombytes(rnd, RNDBYTES);
#else
  for(n=0;n<RNDBYTES;n++)
    rnd[n] = 0;
#endif
  shake256_avx512seq(rhoprime, CRHBYTES, key, SEEDBYTES + RNDBYTES + CRHBYTES);
  /* Expand matrix and transform vectors */
  polyvec_matrix_expand(mat, rho);
 
  polyveck_instailoredntt(&t0);
rej:
  /* Sample intermediate vector y */
  // polyvecl_uniform_gamma1(&y, rhoprime, nonce++);
  #if L == 4
    poly_uniform_gamma1_8x(&y.vec[0], &y.vec[1], &y.vec[2], &y.vec[3], &tmp, &tmp, &tmp, &tmp,rhoprime, nonce, nonce + 1, nonce + 2, nonce + 3, 0, 0, 0, 0);  
    nonce += 4;
  #elif L == 5
    poly_uniform_gamma1_8x(&y.vec[0], &y.vec[1], &y.vec[2], &y.vec[3], &y.vec[4], &tmp, &tmp, &tmp,rhoprime, nonce, nonce + 1, nonce + 2, nonce + 3, nonce + 4, 0, 0, 0);  
    nonce += 5;
  #elif L == 7
    poly_uniform_gamma1_8x(&y.vec[0], &y.vec[1], &y.vec[2], &y.vec[3], &y.vec[4], &y.vec[5], &y.vec[6], &tmp,rhoprime, nonce, nonce + 1, nonce + 2, nonce + 3, nonce + 4, nonce + 5, nonce + 6, 0);  
    nonce += 7;
  #else
  #error
  #endif
  /* Matrix-vector multiplication */
  z = y;
  polyvecl_ntt(&z);
  polyvec_matrix_pointwise_montgomery(&w1, mat, &z);
  polyveck_reduce(&w1);
  polyveck_invntt_tomont(&w1);

  /* Decompose w and call the random oracle */
  polyveck_caddq(&w1);
  polyveck_decompose(&w1, &w0, &w1);
  polyveck_pack_w1(sig, &w1);

  
  uint8_t input2[CRHBYTES+K*POLYW1_PACKEDBYTES];
  memcpy(input2, mu, CRHBYTES);
  memcpy(input2+CRHBYTES, sig, K*POLYW1_PACKEDBYTES);
  shake256_avx512seq(sig, CTILDEBYTES, input2, CRHBYTES+K*POLYW1_PACKEDBYTES);

  poly_challenge(&cp, sig); /* uses only the first SEEDBYTES bytes of sig */

  int i;
  polyveck w0prime;
  #if DILITHIUM_MODE == 2
  if(evaluate_cs1_cs2_early_check_32_AVX512_opt(&z, &w0prime, &cp, s1_table, s2_table, &y, &w0, GAMMA1 - BETA, GAMMA2 - BETA))
  {
    goto rej;
  }
  #elif DILITHIUM_MODE == 3
  if(evaluate_cs2_earlycheck_32_avx512_opt(&w0prime, &w0, &cp, s21_table, s22_table, GAMMA2 - BETA))
  {
    goto rej;
  }
  if(evaluate_cs1_earlycheck_32_avx512_opt(&z, &y, &cp, s11_table, s12_table, GAMMA1 - BETA))
  {
    goto rej;
  }
  #elif DILITHIUM_MODE == 5
  if(evaluate_cs2_earlycheck_AVX512_opt(&w0prime, &cp, s21_table,s22_table, &w0, GAMMA2-BETA))
  {
    goto rej;
  }
  
  if(evaluate_cs1_earlycheck_AVX512_opt(&z, &cp, s11_table, s12_table, &y, GAMMA1-BETA))
  {
    goto rej;
  }
  #endif
  poly_smallntt(&cp);
 
  for(i = 0; i < K ; i ++)
  {
    poly_pointwise_montgomery(&h.vec[i],&cp,&t0.vec[i]);
    poly_invntt_tomont(&h.vec[i]);
    if(poly_chknorm(&h.vec[i], GAMMA2))
    {
      goto rej;
    }
  }
 
  polyveck_add(&w0prime, &w0prime, &h);
  n = polyveck_make_hint(&h, &w0prime, &w1);
  if(n > OMEGA)
    goto rej;
  /* Write signature */
  pack_sig(sig, sig, &z, &h);
  *siglen = CRYPTO_BYTES;
  return 0;
}

/*************************************************
* Name:        crypto_sign
*
* Description: Compute signed message.
*
* Arguments:   - uint8_t *sm: pointer to output signed message (allocated
*                             array with CRYPTO_BYTES + mlen bytes),
*                             can be equal to m
*              - size_t *smlen: pointer to output length of signed
*                               message
*              - const uint8_t *m: pointer to message to be signed
*              - size_t mlen: length of message
*              - const uint8_t *sk: pointer to bit-packed secret key
*
* Returns 0 (success)
**************************************************/
int crypto_sign(uint8_t *sm,
                size_t *smlen,
                const uint8_t *m,
                size_t mlen,
                const uint8_t *sk)
{
  size_t i;

  for(i = 0; i < mlen; ++i)
    sm[CRYPTO_BYTES + mlen - 1 - i] = m[mlen - 1 - i];
  crypto_sign_signature(sm, smlen, sm + CRYPTO_BYTES, mlen, sk);
  *smlen += mlen;
  return 0;
}

/*************************************************
* Name:        crypto_sign_verify
*
* Description: Verifies signature.
*
* Arguments:   - uint8_t *m: pointer to input signature
*              - size_t siglen: length of signature
*              - const uint8_t *m: pointer to message
*              - size_t mlen: length of message
*              - const uint8_t *pk: pointer to bit-packed public key
*
* Returns 0 if signature could be verified correctly and -1 otherwise
**************************************************/
int crypto_sign_verify(const uint8_t *sig,
                       size_t siglen,
                       const uint8_t *m,
                       size_t mlen,
                       const uint8_t *pk)
{
  unsigned int i;
  uint8_t buf[K*POLYW1_PACKEDBYTES];
  uint8_t rho[SEEDBYTES];
  uint8_t mu[CRHBYTES];
  uint8_t c[CTILDEBYTES];
  uint8_t c2[CTILDEBYTES];
  poly cp;
  polyvecl mat[K], z;
  polyveck t1, w1, h;

  if(siglen != CRYPTO_BYTES)
    return -1;

  unpack_pk(rho, &t1, pk);
  if(unpack_sig(c, &z, &h, sig))
    return -1;
  if(polyvecl_chknorm(&z, GAMMA1 - BETA))
    return -1;

  /* Compute CRH(H(rho, t1), msg) */
  shake256_avx512seq(mu, CRHBYTES, pk, CRYPTO_PUBLICKEYBYTES);
  uint8_t input[CRHBYTES+mlen];
  memcpy(input, mu, CRHBYTES);
  memcpy(input+CRHBYTES, m, mlen);
  shake256_avx512seq(mu, CRHBYTES, input, CRHBYTES+mlen);
 

  /* Matrix-vector multiplication; compute Az - c2^dt1 */
  poly_challenge(&cp, c); /* uses only the first SEEDBYTES bytes of c */
  polyvec_matrix_expand(mat, rho);

  polyvecl_ntt(&z);
  polyvec_matrix_pointwise_montgomery(&w1, mat, &z);

  poly_smallntt(&cp);
  polyveck_shiftl(&t1);
  polyveck_ntt(&t1);
  polyveck_pointwise_poly_montgomery(&t1, &cp, &t1);

  polyveck_sub(&w1, &w1, &t1);
  polyveck_reduce(&w1);
  polyveck_invntt_tomont(&w1);

  /* Reconstruct w1 */
  polyveck_caddq(&w1);
  polyveck_use_hint(&w1, &w1, &h);
  polyveck_pack_w1(buf, &w1);

  /* Call random oracle and verify challenge */
  uint8_t input2[CRHBYTES+K*POLYW1_PACKEDBYTES];
  memcpy(input2, mu, CRHBYTES);
  memcpy(input2+CRHBYTES, buf, K*POLYW1_PACKEDBYTES);
  shake256_avx512seq(c2, CTILDEBYTES, input2, CRHBYTES+K*POLYW1_PACKEDBYTES);

  for(i = 0; i < CTILDEBYTES; ++i)
    if(c[i] != c2[i])
      return -1;

  return 0;
}

/*************************************************
* Name:        crypto_sign_open
*
* Description: Verify signed message.
*
* Arguments:   - uint8_t *m: pointer to output message (allocated
*                            array with smlen bytes), can be equal to sm
*              - size_t *mlen: pointer to output length of message
*              - const uint8_t *sm: pointer to signed message
*              - size_t smlen: length of signed message
*              - const uint8_t *pk: pointer to bit-packed public key
*
* Returns 0 if signed message could be verified correctly and -1 otherwise
**************************************************/
int crypto_sign_open(uint8_t *m,
                     size_t *mlen,
                     const uint8_t *sm,
                     size_t smlen,
                     const uint8_t *pk)
{
  size_t i;

  if(smlen < CRYPTO_BYTES)
    goto badsig;

  *mlen = smlen - CRYPTO_BYTES;
  if(crypto_sign_verify(sm, CRYPTO_BYTES, sm + CRYPTO_BYTES, *mlen, pk))
    goto badsig;
  else {
    /* All good, copy msg, return 0 */
    for(i = 0; i < *mlen; ++i)
      m[i] = sm[CRYPTO_BYTES + i];
    return 0;
  }

badsig:
  /* Signature verification failed */
  *mlen = -1;
  for(i = 0; i < smlen; ++i)
    m[i] = 0;

  return -1;
}
