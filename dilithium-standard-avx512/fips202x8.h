#ifndef FIPS202X8_H
#define FIPS202X8_H

#include <immintrin.h>



void keccak_squeezeblocks8x(unsigned char *h0,
                                   unsigned char *h1,
                                   unsigned char *h2,
                                   unsigned char *h3,
                                   unsigned char *h4,
                                   unsigned char *h5,
                                   unsigned char *h6,
                                   unsigned char *h7,
                                   unsigned long long int nblocks,
                                   __m512i *s,
                                   unsigned int r);
void shake128x8(__m512i s[25], unsigned char *out0,
                unsigned char *out1,
                unsigned char *out2,
                unsigned char *out3,
                unsigned char *out4,
                unsigned char *out5,
                unsigned char *out6,
                unsigned char *out7, unsigned long long outlen,
                unsigned char *in0,
                unsigned char *in1,
                unsigned char *in2,
                unsigned char *in3,
                unsigned char *in4,
                unsigned char *in5,
                unsigned char *in6,
                unsigned char *in7, unsigned long long inlen);

void shake256x8(__m512i s[25], unsigned char *out0,
                unsigned char *out1,
                unsigned char *out2,
                unsigned char *out3,
                unsigned char *out4,
                unsigned char *out5,
                unsigned char *out6,
                unsigned char *out7, unsigned long long outlen,
                unsigned char *in0,
                unsigned char *in1,
                unsigned char *in2,
                unsigned char *in3,
                unsigned char *in4,
                unsigned char *in5,
                unsigned char *in6,
                unsigned char *in7, unsigned long long inlen);

#endif
