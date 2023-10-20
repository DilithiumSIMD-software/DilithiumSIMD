#ifndef NTT_H
#define NTT_H

#include <stdint.h>
#include "params.h"
#include <immintrin.h>
#define ntt DILITHIUM_NAMESPACE(_ntt)
void ntt(int32_t a[N]);

#define invntt_tomont DILITHIUM_NAMESPACE(_invntt_tomont)
void tailoredntt_avx(__m512i *a, const __m512i *qdata);
void instailoredntt_avx(__m512i *a, const __m512i *qdata);
void invntt_tomont(int32_t a[N]);
void nttunpack_avx(__m512i *a);
void ntt_avx(__m512i *a, const __m512i *qdata);
void invntt_avx(__m512i *a, const __m512i *qdata);
void pointwise_avx(__m512i *c, const __m512i *a, const __m512i *b, const __m512i *qdata);
void smallntt_avx(__m512i *a, const __m512i *qdata);
void pointwise_acc_avx(__m512i *c, const __m512i *a, const __m512i *b, const __m512i *qdata);



#endif
