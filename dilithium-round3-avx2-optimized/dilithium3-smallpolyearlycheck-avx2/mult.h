#include <stdio.h>
#include <stdint.h>
#include "params.h"
#include "poly.h"
#include "polyvec.h"

void prepare_s1_table(uint64_t s_table[2*N], const polyvecl *s1);
void evaluate_cs1(polyvecl *z, const poly *c, const uint64_t s_table[2*N]);
void prepare_s2_table(uint64_t s2_table[2*N],const polyveck *s2);
void evaluate_cs2(
		polyveck *z, 
		poly *c, const uint64_t s2_table[2*N]);
void prepare_t0_table(uint64_t t00_table[4*N], uint64_t t01_table[4*N],const polyveck *t0);
void evaluate_ct0(polyveck *z,const poly *c, 
                    const uint64_t t00_table[4*N], const uint64_t t01_table[4*N]);
void prepare_t1_table(uint64_t t10_table[4*N], uint64_t t11_table[4*N],const polyveck *t1);
void evaluate_ct1(polyveck *z, const poly *c, 
		const uint64_t t10_table[4*N], const uint64_t t11_table[4*N]);
int evaluate_cs2_earlycheck(
		polyveck *z, polyveck *w0,
		poly *c, const uint64_t s2_table[2*N], int32_t B);
int evaluate_cs1_earlycheck(polyvecl *z, polyvecl *y, const poly *c, const uint64_t s_table[2*N], int32_t B);
void prepare_s1_table_32(uint32_t s11_table[2*N], uint32_t s12_table[2*N],const polyvecl *s1);
int evaluate_cs1_earlycheck_32(polyvecl *z, polyvecl *y, const poly *c, const uint32_t s11_table[2*N], const uint32_t s12_table[2*N], int32_t B);
void prepare_s2_table_32(uint32_t s21_table[2*N], uint32_t s22_table[2*N],const polyveck *s2);
int evaluate_cs2_earlycheck_32(
		polyveck *z, polyveck *w0,
		poly *c, uint32_t s21_table[2*N], uint32_t s22_table[2*N], int32_t B);
void prepare_s1_table_32_avx2(uint32_t s11_table[2*N], uint32_t s12_table[2*N],const polyvecl *s1);
void prepare_s2_table_32_avx2(uint32_t s21_table[2*N], uint32_t s22_table[2*N],const polyveck *s2);
int evaluate_cs1_earlycheck_32_avx2(polyvecl *z, polyvecl *y, const poly *c, const uint32_t s11_table[2*N], const uint32_t s12_table[2*N], int32_t B);
void add_asm(uint32_t *answer, const uint32_t *s_table);
void addmask11_asm(uint32_t *answer, const uint32_t *s_table);
void addmask12_asm(uint32_t *answer, const uint32_t *s_table);
int evaluate_cs2_earlycheck_32_avx2(
		polyveck *z, polyveck *w0,
		poly *c, uint32_t s21_table[2*N], uint32_t s22_table[2*N], int32_t B);
// int evaluate_cs1_earlycheck_32_avx2(polyvecl *z, polyvecl *y, const poly *c, const uint32_t s11_table[2*N], const uint32_t s12_table[2*N], int32_t B)
