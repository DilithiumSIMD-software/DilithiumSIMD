#include <stdio.h>
#include <stdint.h>
#include "params.h"
#include "poly.h"
#include "polyvec.h"

void prepare_s1_table(uint32_t s1_table[2*N], polyvecl *s1);
void prepare_s2_table(uint32_t s2_table[2*N], polyveck *s2);
int evaluate_cs1_cs2_early_check_32_AVX512_opt(polyvecl *z1, polyveck *z2, const poly *c, const uint32_t s1_table[2*N], const uint32_t s2_table[2*N], polyvecl *y, polyveck *w0, int32_t A, int32_t B);
