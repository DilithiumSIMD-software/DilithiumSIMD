#include <stdio.h>
#include <stdint.h>
#include "params.h"
#include "poly.h"
#include "polyvec.h"


int evaluate_cs1_cs2_early_check_32_asm(polyvecl *z1, polyveck *z2, const poly *c, const uint32_t s1_table[2*N], const uint32_t s2_table[2*N], polyvecl *y, polyveck *w0, int32_t A, int32_t B);


void prepare_s1_table(uint32_t s1_table[2*N], polyvecl *s1);
void prepare_s2_table(uint32_t s2_table[2*N], polyveck *s2);

void add_asm(uint32_t *answer, const uint32_t *s_table);
void addmask_asm(uint32_t *answer, const uint32_t *s_table);
void prepare_t0_table(uint64_t t00_table[2*N], uint64_t t01_table[2*N], polyveck *t0);
void evaluate_ct0(polyveck *z,const poly *c, 
                    const uint64_t t00_table[2*N], const uint64_t t01_table[2*N]);
