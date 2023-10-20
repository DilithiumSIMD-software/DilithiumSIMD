#include "consts.h"


.global cdecl(shufflelo8_avx)
cdecl(shufflelo8_avx):
vmovdqa32	(%rsi),%zmm1
vmovdqa32	(%rdx),%zmm2
vshufi32x4	$0x44,%zmm2,%zmm1,%zmm3
vmovdqa32	%zmm3,(%rdi)


ret

.global cdecl(shufflehi8_avx)
cdecl(shufflehi8_avx):
vmovdqa32	(%rsi),%zmm1
vmovdqa32	(%rdx),%zmm2
vshufi32x4	$0xEE,%zmm2,%zmm1,%zmm3
vmovdqa32	%zmm3,(%rdi)


ret