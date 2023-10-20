#include "consts.h"
.include "shuffle.inc"

.text
nttunpack256_avx:
#load
vmovdqa32		  0(%rdi),%zmm1
vmovdqa32		 64(%rdi),%zmm2
vmovdqa32		128(%rdi),%zmm3
vmovdqa32	 	192(%rdi),%zmm4
vmovdqa32		256(%rdi),%zmm5
vmovdqa32		320(%rdi),%zmm6
vmovdqa32		384(%rdi),%zmm7
vmovdqa32	 	448(%rdi),%zmm8
vmovdqa32		512(%rdi),%zmm9
vmovdqa32		576(%rdi),%zmm10
vmovdqa32		640(%rdi),%zmm11
vmovdqa32	 	704(%rdi),%zmm12
vmovdqa32		768(%rdi),%zmm13
vmovdqa32		832(%rdi),%zmm14
vmovdqa32		896(%rdi),%zmm15
vmovdqa32	 	960(%rdi),%zmm16

shuffle8	1,2,22,2
shuffle8	3,4,1,4
shuffle8	5,6,3,6
shuffle8	7,8,5,8
shuffle8	9,10,7,10
shuffle8	11,12,9,12
shuffle8	13,14,11,14
shuffle8	15,16,13,16

shuffle4	22,2,15,28
shuffle4	1,4,22,2
shuffle4	3,6,1,4
shuffle4	5,8,3,6
shuffle4	7,10,5,8
shuffle4	9,12,7,10
shuffle4	11,14,9,12
shuffle4	13,16,11,14

shuffle2	15,28,13,16
shuffle2	22,2,15,28
shuffle2	1,4,22,2
shuffle2	3,6,1,4
shuffle2	5,8,3,6
shuffle2	7,10,5,8
shuffle2	9,12,7,10
shuffle2	11,14,9,12

shuffle1	13,16,11,14
shuffle1	15,28,13,16
shuffle1	22,2,15,28
shuffle1	1,4,22,2
shuffle1	3,6,1,4
shuffle1	5,8,3,6
shuffle1	7,10,5,8
shuffle1	9,12,7,10


#store
vmovdqa32		%zmm11,  0(%rdi)
vmovdqa32		%zmm14, 64(%rdi)
vmovdqa32		%zmm13,128(%rdi)
vmovdqa32	 	%zmm16,192(%rdi)
vmovdqa32		%zmm15,256(%rdi)
vmovdqa32		%zmm28,320(%rdi)
vmovdqa32		%zmm22,384(%rdi)
vmovdqa32	 	%zmm2,448(%rdi)
vmovdqa32		%zmm1 ,512(%rdi)
vmovdqa32		%zmm4,576(%rdi)
vmovdqa32		%zmm3,640(%rdi)
vmovdqa32	 	%zmm6,704(%rdi)
vmovdqa32		%zmm5,768(%rdi)
vmovdqa32		%zmm8,832(%rdi)
vmovdqa32		%zmm7,896(%rdi)
vmovdqa32	 	%zmm10,960(%rdi)

ret

.global cdecl(nttunpack_avx)
cdecl(nttunpack_avx):
mov     $0xAAAA, %eax
kmovw   %eax, %k7
mov     $0x0F0F, %eax
kmovw   %eax, %k6
call		nttunpack256_avx
ret
