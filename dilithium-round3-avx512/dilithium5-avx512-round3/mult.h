#include <stdio.h>
#include <stdint.h>
#include "params.h"
#include "poly.h"
#include "polyvec.h"
//32bit 拼满
void prepare_s_table4x(uint32_t s_table[2*N],  poly *a);
//32bit 只拼其中的24bit
void prepare_s_table3x(uint32_t s_table[2*N],  poly *a);

int evaluate_cs2_earlycheck_AVX512_opt(polyveck *z2,  const poly *c, const uint32_t s21_table[2*N], const uint32_t s22_table[2*N],  polyveck *w0, int32_t B);
int evaluate_cs1_earlycheck_AVX512_opt(polyvecl *z,  const poly *c, const uint32_t s11_table[2*N], const uint32_t s12_table[2*N],  polyvecl *y,  int32_t A);