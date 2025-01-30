#ifndef ROUNDING_H
#define ROUNDING_H

#include <stdint.h>
#include "params.h"
#include <immintrin.h>

#define power2round DILITHIUM_NAMESPACE(_power2round)
int32_t power2round(int32_t *a0, int32_t a);

#define decompose DILITHIUM_NAMESPACE(_decompose)
int32_t decompose(int32_t *a0, int32_t a);

#define make_hint DILITHIUM_NAMESPACE(_make_hint)
unsigned int make_hint(int32_t a0, int32_t a1);

#define use_hint DILITHIUM_NAMESPACE(_use_hint)
int32_t use_hint(int32_t a, unsigned int hint);

void power2round_avx(__m512i *a1, __m512i *a0, const __m512i *a);
void decompose_avx(__m512i *a1, __m512i *a0, const __m512i *a);
unsigned int make_hint_avx(int32_t * restrict h, const int32_t * restrict a0, const int32_t * restrict a1);
void use_hint_avx(__m512i *b, const __m512i *a, const __m512i * restrict hint);

#endif
