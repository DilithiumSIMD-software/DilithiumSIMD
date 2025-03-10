#include <stdio.h>
#include <stdint.h>
#include "params.h"
#include "poly.h"
#include "polyvec.h"
void prepare_s_table4x(uint32_t s_table[2*N],  poly *a);
void prepare_s_table3x(uint32_t s_table[2*N],  poly *a);

int evaluate_cs2_earlycheck_AVX512_opt(polyveck *z2,  const poly *c, const uint32_t s21_table[2*N], const uint32_t s22_table[2*N],  polyveck *w0, int32_t B);
int evaluate_cs1_earlycheck_AVX512_opt(polyvecl *z,  const poly *c, const uint32_t s11_table[2*N], const uint32_t s12_table[2*N],  polyvecl *y,  int32_t A);


int cs2_earlycheck_AVX512(polyveck *z2, const poly *c, int8_t s2table[2*K*N], polyveck *w0, int32_t B);