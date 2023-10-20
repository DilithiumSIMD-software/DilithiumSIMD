#include <stdio.h>
#include <stdint.h>
#include "params.h"
#include "poly.h"
#include "polyvec.h"
void prepare_s1_table_32_avx512(uint32_t s11_table[2*N], uint32_t s12_table[2*N],const polyvecl *s1);
void prepare_s2_table_32_avx512(uint32_t s21_table[2*N], uint32_t s22_table[2*N],const polyveck *s2);
int evaluate_cs1_earlycheck_32_avx512_opt(polyvecl *z, polyvecl *y, const poly *c, const uint32_t s11_table[2*N], const uint32_t s12_table[2*N], int32_t B);
int evaluate_cs2_earlycheck_32_avx512_opt(polyveck *z, polyveck *w0,const poly *c, const uint32_t s21_table[2*N], const uint32_t s22_table[2*N], int32_t B);