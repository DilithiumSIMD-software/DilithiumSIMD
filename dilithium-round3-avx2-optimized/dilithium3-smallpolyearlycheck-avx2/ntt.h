#ifndef NTT_H
#define NTT_H

#include <stdint.h>
#include "params.h"

#define ntt_avx DILITHIUM_NAMESPACE(_ntt_avx)
void ntt_avx(int32_t a[N], const int32_t *qdata);
#define ntt_avx_lazy_reduction DILITHIUM_NAMESPACE(_ntt_avx_lazy_reduction)
void ntt_avx_lazy_reduction(int32_t a[N], const int32_t *qdata);

#define ntt_avx_tailored_reduction DILITHIUM_NAMESPACE(_ntt_avx_tailored_reduction)
void ntt_avx_tailored_reduction(int32_t a[N], const int32_t *qdata);

#define invntt_avx DILITHIUM_NAMESPACE(_invntt_avx)
void invntt_avx(int32_t a[N], const int32_t *qdata);

#define pointwise_avx DILITHIUM_NAMESPACE(_pointwise_avx)
void pointwise_avx(int32_t c[N],
                   const int32_t a[N],
                   const int32_t b[N],
                   const int32_t *qdata);
#define pointwise_acc_avx DILITHIUM_NAMESPACE(_pointwise_acc_avx)
void pointwise_acc_avx(int32_t c[N],
                       const int32_t *a,
                       const int32_t *b,
                       const int32_t *qdata);

#endif


