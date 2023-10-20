#include "params.h"
#include "consts.h"


.macro mulmontreduce rh0,rh1,zh0
vpsrlq		$32,%zmm\rh0,%zmm28
vpsrlq		$32,%zmm\rh1,%zmm29
#mul
vpmuldq		%zmm\rh0,%zmm\rh1,%zmm\rh0
vpmuldq		%zmm28,%zmm29,%zmm28

#reduce
vpmuldq		%zmm30,%zmm\rh0,%zmm\rh1
vpmuldq		%zmm30,%zmm28,%zmm29
vpmuldq		%zmm0,%zmm\rh1,%zmm\rh1
vpmuldq		%zmm0,%zmm29,%zmm29
vpsubq		%zmm\rh1,%zmm\rh0,%zmm\rh0
vpsubq		%zmm29,%zmm28,%zmm28
vpsrlq		$32,%zmm\rh0,%zmm\rh0
vpblendmd	%zmm28,%zmm\rh0,%zmm\zh0{%k7}
.endm

.macro montreduce rh0,rh1,zh0
#reduce
vpmuldq		%zmm30,%zmm\rh0,%zmm27
vpmuldq		%zmm30,%zmm\rh1,%zmm28
vpmuldq		%zmm0,%zmm27,%zmm27
vpmuldq		%zmm0,%zmm28,%zmm28
vpsubq		%zmm27,%zmm\rh0,%zmm27
vpsubq		%zmm28,%zmm\rh1,%zmm28
vpsrlq		$32,%zmm27,%zmm27
vpblendmd	%zmm28,%zmm27,%zmm\zh0{%k7}
.endm

.macro mult rh0,rh1,rl0,rl1
vpsrlq		$32,%zmm\rh0,%zmm28
vpsrlq		$32,%zmm\rh1,%zmm29
#mul
vpmuldq		%zmm\rh0,%zmm\rh1,%zmm\rl0
vpmuldq		%zmm28,%zmm29,%zmm\rl1
.endm

.macro polypointwise off
vmovdqa32	( 0 +\off)(%rsi),%zmm1
vmovdqa32   (64 +\off)(%rsi),%zmm2
vmovdqa32	(128+\off)(%rsi),%zmm3
vmovdqa32	(192+\off)(%rsi),%zmm4
vmovdqa32	(256+\off)(%rsi),%zmm5
vmovdqa32	(320+\off)(%rsi),%zmm6
vmovdqa32	(384+\off)(%rsi),%zmm7
vmovdqa32	(448+\off)(%rsi),%zmm8
vmovdqa32	( 0 +\off)(%rdx),%zmm9
vmovdqa32	(64 +\off)(%rdx),%zmm10
vmovdqa32	(128+\off)(%rdx),%zmm11
vmovdqa32	(192+\off)(%rdx),%zmm12
vmovdqa32	(256+\off)(%rdx),%zmm13
vmovdqa32	(320+\off)(%rdx),%zmm14
vmovdqa32	(384+\off)(%rdx),%zmm15
vmovdqa32	(448+\off)(%rdx),%zmm16

mulmontreduce  1,9,17
mulmontreduce  2,10,18
mulmontreduce  3,11,19
mulmontreduce  4,12,20
mulmontreduce  5,13,21
mulmontreduce  6,14,22
mulmontreduce  7,15,23
mulmontreduce  8,16,24

vmovdqa32	%zmm17,( 0 +\off)(%rdi)
vmovdqa32   %zmm18,(64 +\off)(%rdi)
vmovdqa32	%zmm19,(128+\off)(%rdi)
vmovdqa32	%zmm20,(192+\off)(%rdi)
vmovdqa32	%zmm21,(256+\off)(%rdi)
vmovdqa32	%zmm22,(320+\off)(%rdi)
vmovdqa32	%zmm23,(384+\off)(%rdi)
vmovdqa32	%zmm24,(448+\off)(%rdi)
.endm

.text
.global cdecl(pointwise_avx)
cdecl(pointwise_avx):
#consts
vmovdqa32	_16XQ*4(%rcx),%zmm0
vmovdqa32	_16XQINV*4(%rcx),%zmm30
mov     $0xAAAA, %eax
kmovw   %eax, %k7
polypointwise 0
polypointwise 512

ret


.macro acc rh0,rh1,rh2,rh3,rh4,rh5,rh6,rh7
vpaddq		%zmm5,%zmm\rh0,%zmm5
vpaddq		%zmm6,%zmm\rh1,%zmm6
vpaddq		%zmm7,%zmm\rh2,%zmm7
vpaddq		%zmm8,%zmm\rh3,%zmm8
vpaddq		%zmm17,%zmm\rh4,%zmm17
vpaddq		%zmm18,%zmm\rh5,%zmm18
vpaddq		%zmm19,%zmm\rh6,%zmm19
vpaddq		%zmm20,%zmm\rh7,%zmm20
.endm

.macro polyvecpointwise off
vmovdqa32	( 0 +\off)(%rsi),%zmm1
vmovdqa32   (64 +\off)(%rsi),%zmm2
vmovdqa32	(128+\off)(%rsi),%zmm3
vmovdqa32	(192+\off)(%rsi),%zmm4
vmovdqa32	( 0 +\off)(%rdx),%zmm9
vmovdqa32	(64 +\off)(%rdx),%zmm10
vmovdqa32	(128+\off)(%rdx),%zmm11
vmovdqa32	(192+\off)(%rdx),%zmm12

# a0*s0
mult  1,9,5,6
mult  2,10,7,8
mult  3,11,17,18
mult  4,12,19,20


# a1*s1
vmovdqa32	( 0 +1024+\off)(%rsi),%zmm1
vmovdqa32   (64 +1024+\off)(%rsi),%zmm2
vmovdqa32	(128+1024+\off)(%rsi),%zmm3
vmovdqa32	(192+1024+\off)(%rsi),%zmm4
vmovdqa32	( 0 +1024+\off)(%rdx),%zmm9
vmovdqa32	(64 +1024+\off)(%rdx),%zmm10
vmovdqa32	(128+1024+\off)(%rdx),%zmm11
vmovdqa32	(192+1024+\off)(%rdx),%zmm12

mult  1,9,1,9
mult  2,10,2,10
mult  3,11,3,11
mult  4,12,4,12

acc 1,9,2,10,3,11,4,12

#if L >= 3
# a1*s1
vmovdqa32	( 0 +2048+\off)(%rsi),%zmm1
vmovdqa32   (64 +2048+\off)(%rsi),%zmm2
vmovdqa32	(128+2048+\off)(%rsi),%zmm3
vmovdqa32	(192+2048+\off)(%rsi),%zmm4
vmovdqa32	( 0 +2048+\off)(%rdx),%zmm9
vmovdqa32	(64 +2048+\off)(%rdx),%zmm10
vmovdqa32	(128+2048+\off)(%rdx),%zmm11
vmovdqa32	(192+2048+\off)(%rdx),%zmm12

mult  1,9,1,9
mult  2,10,2,10
mult  3,11,3,11
mult  4,12,4,12

acc 1,9,2,10,3,11,4,12
#endif

#if L >= 4

vmovdqa32	( 0 +3072+\off)(%rsi),%zmm1
vmovdqa32   (64 +3072+\off)(%rsi),%zmm2
vmovdqa32	(128+3072+\off)(%rsi),%zmm3
vmovdqa32	(192+3072+\off)(%rsi),%zmm4
vmovdqa32	( 0 +3072+\off)(%rdx),%zmm9
vmovdqa32	(64 +3072+\off)(%rdx),%zmm10
vmovdqa32	(128+3072+\off)(%rdx),%zmm11
vmovdqa32	(192+3072+\off)(%rdx),%zmm12

mult  1,9,1,9
mult  2,10,2,10
mult  3,11,3,11
mult  4,12,4,12

acc 1,9,2,10,3,11,4,12

#endif

#if L >= 5

vmovdqa32	( 0 +4096+\off)(%rsi),%zmm1
vmovdqa32   (64 +4096+\off)(%rsi),%zmm2
vmovdqa32	(128+4096+\off)(%rsi),%zmm3
vmovdqa32	(192+4096+\off)(%rsi),%zmm4
vmovdqa32	( 0 +4096+\off)(%rdx),%zmm9
vmovdqa32	(64 +4096+\off)(%rdx),%zmm10
vmovdqa32	(128+4096+\off)(%rdx),%zmm11
vmovdqa32	(192+4096+\off)(%rdx),%zmm12

mult  1,9,1,9
mult  2,10,2,10
mult  3,11,3,11
mult  4,12,4,12

acc 1,9,2,10,3,11,4,12
#endif

#if L >= 6

vmovdqa32	( 0 +5120+\off)(%rsi),%zmm1
vmovdqa32   (64 +5120+\off)(%rsi),%zmm2
vmovdqa32	(128+5120+\off)(%rsi),%zmm3
vmovdqa32	(192+5120+\off)(%rsi),%zmm4
vmovdqa32	( 0 +5120+\off)(%rdx),%zmm9
vmovdqa32	(64 +5120+\off)(%rdx),%zmm10
vmovdqa32	(128+5120+\off)(%rdx),%zmm11
vmovdqa32	(192+5120+\off)(%rdx),%zmm12

mult  1,9,1,9
mult  2,10,2,10
mult  3,11,3,11
mult  4,12,4,12

acc 1,9,2,10,3,11,4,12
#endif

#if L >= 7

vmovdqa32	( 0 +6144+\off)(%rsi),%zmm1
vmovdqa32   (64 +6144+\off)(%rsi),%zmm2
vmovdqa32	(128+6144+\off)(%rsi),%zmm3
vmovdqa32	(192+6144+\off)(%rsi),%zmm4
vmovdqa32	( 0 +6144+\off)(%rdx),%zmm9
vmovdqa32	(64 +6144+\off)(%rdx),%zmm10
vmovdqa32	(128+6144+\off)(%rdx),%zmm11
vmovdqa32	(192+6144+\off)(%rdx),%zmm12

mult  1,9,1,9
mult  2,10,2,10
mult  3,11,3,11
mult  4,12,4,12

acc 1,9,2,10,3,11,4,12
#endif

montreduce  5,6,1
montreduce  7,8,2
montreduce  17,18,3
montreduce  19,20,4


vmovdqa32	%zmm1,( 0 +\off)(%rdi)
vmovdqa32   %zmm2,(64 +\off)(%rdi)
vmovdqa32	%zmm3,(128+\off)(%rdi)
vmovdqa32	%zmm4,(192+\off)(%rdi)
.endm



.global cdecl(pointwise_acc_avx)
cdecl(pointwise_acc_avx):
vmovdqa32	_16XQ*4(%rcx),%zmm0
vmovdqa32	_16XQINV*4(%rcx),%zmm30
mov     $0xAAAA, %eax
kmovw   %eax, %k7

polyvecpointwise  0
polyvecpointwise  256
polyvecpointwise  512
polyvecpointwise  768

ret