#include "immintrin.h"
#include "emmintrin.h"
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#define ALIGN __attribute__ ((aligned (32)))

int keccak_avx512(const uint8_t *in, int inlen, uint8_t *md, int r);
void keccakF(__m512i* x, int rnd);
