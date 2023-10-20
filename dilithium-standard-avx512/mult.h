#include <stdio.h>
#include <stdint.h>
#include "params.h"
#include "poly.h"
#include "polyvec.h"

#if DILITHIUM_MODE == 2
void prepare_s1_table(uint32_t s1_table[2*N], polyvecl *s1);
void prepare_s2_table(uint32_t s2_table[2*N], polyveck *s2);
int evaluate_cs1_cs2_early_check_32_AVX512_opt(polyvecl *z1, polyveck *z2, const poly *c, const uint32_t s1_table[2*N], const uint32_t s2_table[2*N], polyvecl *y, polyveck *w0, int32_t A, int32_t B);

#elif DILITHIUM_MODE == 3
void prepare_s1_table_32_avx512(uint32_t s11_table[2*N], uint32_t s12_table[2*N],const polyvecl *s1);
void prepare_s2_table_32_avx512(uint32_t s21_table[2*N], uint32_t s22_table[2*N],const polyveck *s2);
int evaluate_cs1_earlycheck_32_avx512_opt(polyvecl *z, polyvecl *y, const poly *c, const uint32_t s11_table[2*N], const uint32_t s12_table[2*N], int32_t B);
int evaluate_cs2_earlycheck_32_avx512_opt(polyveck *z, polyveck *w0, poly *c, uint32_t s21_table[2*N], uint32_t s22_table[2*N], int32_t B);
#elif DILITHIUM_MODE == 5
void prepare_s_table4x(uint32_t s_table[2*N],  poly *a);
void prepare_s_table3x(uint32_t s_table[2*N],  poly *a);
int evaluate_cs2_earlycheck_AVX512_opt(polyveck *z2,  const poly *c, const uint32_t s21_table[2*N], const uint32_t s22_table[2*N],  polyveck *w0, int32_t B);
int evaluate_cs1_earlycheck_AVX512_opt(polyvecl *z,  const poly *c, const uint32_t s11_table[2*N], const uint32_t s12_table[2*N],  polyvecl *y,  int32_t A);
#endif