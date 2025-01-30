#ifndef ROUNDING_H
#define ROUNDING_H

#include <stdint.h>
#include "params.h"
#include <immintrin.h>
#include "poly.h"

void power2round_avx(__m512i *a1, __m512i *a0, const __m512i *a);
void decompose_avx(__m512i *a1, __m512i *a0, const __m512i *a);
unsigned int make_hint_avx(int32_t * restrict h, const int32_t * restrict a0, const int32_t * restrict a1);
void use_hint_avx(__m512i *b, const __m512i *a, const __m512i * restrict hint);

#endif
