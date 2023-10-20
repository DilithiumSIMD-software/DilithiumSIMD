#include <stdio.h>
#include <stdint.h>
#include "params.h"
#include "poly.h"
#include "polyvec.h"
int evaluate_cs2_earlycheck_avx2(polyveck *z2,  const poly *c, const uint32_t s21_table[2*N], const uint32_t s22_table[2*N],  polyveck *w0, int32_t B);
int evaluate_cs1_earlycheck_avx2(polyvecl *z,  const poly *c, const uint32_t s11_table[2*N], const uint32_t s12_table[2*N],  polyvecl *y,  int32_t A);
void add_asm(uint32_t *answer, const uint32_t *s_table);
void addmask_asm(uint32_t *answer, const uint32_t *s_table);
void addmask_asm2(uint32_t *answer, const uint32_t *s_table);
void prepare_s1_table_avx2(uint32_t s11_table[2*N],uint32_t s12_table[2*N],polyvecl *s1);
void prepare_s2_table_avx2(uint32_t s21_table[2*N],uint32_t s22_table[2*N],polyveck *s2);
void prepare_t0_table(uint64_t t00_table[4*N], uint64_t t01_table[4*N],uint64_t t02_table[4*N], polyveck *t0);
void evaluate_ct0(
		polyveck *z,const poly *c,
		const uint64_t t00_table[4*N], const uint64_t t01_table[4*N], const uint64_t t02_table[4*N]);
