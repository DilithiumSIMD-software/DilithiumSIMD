#include "immintrin.h"
#include "emmintrin.h"
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#define ALIGN __attribute__ ((aligned (32)))
int keccak(char **in_e, int inlen, uint8_t **md, int ra, int out);
void keccakF8x(__m512i state[25]);