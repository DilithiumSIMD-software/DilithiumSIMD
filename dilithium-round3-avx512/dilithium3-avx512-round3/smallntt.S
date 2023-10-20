#include "consts.h"
.include "shuffle.inc"

.macro butterfly l,h,zl0=17,zl1=17,zh0=18,zh1=18
 
vpmuldq		%zmm\zl0,%zmm\h,%zmm20
vmovshdup	%zmm\h,%zmm19
vpmuldq		%zmm\zl1,%zmm19,%zmm21
 
vpmuldq		%zmm0,%zmm20,%zmm20
vpmuldq		%zmm0,%zmm21,%zmm21
 
vpmuldq		%zmm\zh0,%zmm\h,%zmm\h
vpmuldq		%zmm\zh1,%zmm19,%zmm19
 
vpsubq      %zmm20,%zmm\h,%zmm20
vpsubq      %zmm21,%zmm19,%zmm21
vmovshdup	%zmm20,%zmm20
vpblendmd	%zmm21,%zmm20,%zmm21{%k7}
 
vpsubd		%zmm21,%zmm\l,%zmm\h
 
vpaddd		%zmm21,%zmm\l,%zmm\l
.endm

.macro smallbutterfly l,h,zh0=18
vpmulld		%zmm\zh0,%zmm\h,%zmm20
vpsubd		%zmm20,%zmm\l,%zmm\h
vpaddd		%zmm20,%zmm\l,%zmm\l
.endm

.macro levels0t7 
/* level 0 */
vpbroadcastd	(_ZETAS+0)*4(%rsi),%zmm18
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

smallbutterfly	1,9
smallbutterfly	2,10
smallbutterfly	3,11
smallbutterfly	4,12
smallbutterfly	5,13
smallbutterfly	6,14
smallbutterfly	7,15
smallbutterfly	8,16

/* level 1 */
vpbroadcastd	(_ZETAS_QINV+2)*4(%rsi),%zmm17
vpbroadcastd	(_ZETAS+2)*4(%rsi),%zmm18
butterfly	1,5
butterfly	2,6
butterfly	3,7
butterfly	4,8


vpbroadcastd	(_ZETAS_QINV+3)*4(%rsi),%zmm17
vpbroadcastd	(_ZETAS+3)*4(%rsi),%zmm18
butterfly	9,13
butterfly	10,14
butterfly	11,15
butterfly	12,16

/* level 2 */
vpbroadcastd	(_ZETAS_QINV+4)*4(%rsi),%zmm17
vpbroadcastd	(_ZETAS+4)*4(%rsi),%zmm18

butterfly	1,3
butterfly	2,4

vpbroadcastd	(_ZETAS_QINV+5)*4(%rsi),%zmm17
vpbroadcastd	(_ZETAS+5)*4(%rsi),%zmm18

butterfly	5,7
butterfly	6,8

vpbroadcastd	(_ZETAS_QINV+6)*4(%rsi),%zmm17
vpbroadcastd	(_ZETAS+6)*4(%rsi),%zmm18

butterfly	9,11
butterfly	10,12

vpbroadcastd	(_ZETAS_QINV+7)*4(%rsi),%zmm17
vpbroadcastd	(_ZETAS+7)*4(%rsi),%zmm18

butterfly	13,15
butterfly	14,16


/* level 3 */
vpbroadcastd		(_ZETAS_QINV+8)*4(%rsi),%zmm17
vpbroadcastd		(_ZETAS+8)*4(%rsi),%zmm18

butterfly	1,2

vpbroadcastd		(_ZETAS_QINV+9)*4(%rsi),%zmm17
vpbroadcastd		(_ZETAS+9)*4(%rsi),%zmm18

butterfly	3,4

vpbroadcastd		(_ZETAS_QINV+10)*4(%rsi),%zmm17
vpbroadcastd		(_ZETAS+10)*4(%rsi),%zmm18

butterfly	5,6

vpbroadcastd		(_ZETAS_QINV+11)*4(%rsi),%zmm17
vpbroadcastd		(_ZETAS+11)*4(%rsi),%zmm18

butterfly	7,8

vpbroadcastd		(_ZETAS_QINV+12)*4(%rsi),%zmm17
vpbroadcastd		(_ZETAS+12)*4(%rsi),%zmm18

butterfly	9,10

vpbroadcastd		(_ZETAS_QINV+13)*4(%rsi),%zmm17
vpbroadcastd		(_ZETAS+13)*4(%rsi),%zmm18

butterfly	11,12

vpbroadcastd		(_ZETAS_QINV+14)*4(%rsi),%zmm17
vpbroadcastd		(_ZETAS+14)*4(%rsi),%zmm18

butterfly	13,14

vpbroadcastd		(_ZETAS_QINV+15)*4(%rsi),%zmm17
vpbroadcastd		(_ZETAS+15)*4(%rsi),%zmm18

butterfly	15,16

/* level 4 */
shuffle8	1,2,22,2
shuffle8	3,4,1,4
shuffle8	5,6,3,6
shuffle8	7,8,5,8
shuffle8	9,10,7,10
shuffle8	11,12,9,12
shuffle8	13,14,11,14
shuffle8	15,16,13,16

vmovdqa32		(_ZETAS_QINV+16)*4(%rsi),%zmm17
vmovdqa32		(_ZETAS+16)*4(%rsi),%zmm18

butterfly	22,2

vmovdqa32		(_ZETAS_QINV+32)*4(%rsi),%zmm17
vmovdqa32		(_ZETAS+32)*4(%rsi),%zmm18

butterfly	1,4

vmovdqa32		(_ZETAS_QINV+48)*4(%rsi),%zmm17
vmovdqa32		(_ZETAS+48)*4(%rsi),%zmm18

butterfly	3,6

vmovdqa32		(_ZETAS_QINV+64)*4(%rsi),%zmm17
vmovdqa32		(_ZETAS+64)*4(%rsi),%zmm18

butterfly	5,8

vmovdqa32		(_ZETAS_QINV+80)*4(%rsi),%zmm17
vmovdqa32		(_ZETAS+80)*4(%rsi),%zmm18

butterfly	7,10

vmovdqa32		(_ZETAS_QINV+96)*4(%rsi),%zmm17
vmovdqa32		(_ZETAS+96)*4(%rsi),%zmm18

butterfly	9,12

vmovdqa32		(_ZETAS_QINV+112)*4(%rsi),%zmm17
vmovdqa32		(_ZETAS+112)*4(%rsi),%zmm18

butterfly	11,14

vmovdqa32		(_ZETAS_QINV+128)*4(%rsi),%zmm17
vmovdqa32		(_ZETAS+128)*4(%rsi),%zmm18

butterfly	13,16

/* level 5 */
shuffle4	22,2,15,28
shuffle4	1,4,22,2
shuffle4	3,6,1,4
shuffle4	5,8,3,6
shuffle4	7,10,5,8
shuffle4	9,12,7,10
shuffle4	11,14,9,12
shuffle4	13,16,11,14


vmovdqa32		(_ZETAS_QINV+144)*4(%rsi),%zmm17
vmovdqa32		(_ZETAS+144)*4(%rsi),%zmm18

butterfly	15,28

vmovdqa32		(_ZETAS_QINV+160)*4(%rsi),%zmm17
vmovdqa32		(_ZETAS+160)*4(%rsi),%zmm18

butterfly	22,2

vmovdqa32		(_ZETAS_QINV+176)*4(%rsi),%zmm17
vmovdqa32		(_ZETAS+176)*4(%rsi),%zmm18

butterfly	1,4

vmovdqa32		(_ZETAS_QINV+192)*4(%rsi),%zmm17
vmovdqa32		(_ZETAS+192)*4(%rsi),%zmm18

butterfly	3,6

vmovdqa32		(_ZETAS_QINV+208)*4(%rsi),%zmm17
vmovdqa32		(_ZETAS+208)*4(%rsi),%zmm18

butterfly	5,8

vmovdqa32		(_ZETAS_QINV+224)*4(%rsi),%zmm17
vmovdqa32		(_ZETAS+224)*4(%rsi),%zmm18

butterfly	7,10

vmovdqa32		(_ZETAS_QINV+240)*4(%rsi),%zmm17
vmovdqa32		(_ZETAS+240)*4(%rsi),%zmm18

butterfly	9,12

vmovdqa32		(_ZETAS_QINV+256)*4(%rsi),%zmm17
vmovdqa32		(_ZETAS+256)*4(%rsi),%zmm18

butterfly	11,14

/* level 6 */
shuffle2	15,28,13,16
shuffle2	22,2,15,28
shuffle2	1,4,22,2
shuffle2	3,6,1,4
shuffle2	5,8,3,6
shuffle2	7,10,5,8
shuffle2	9,12,7,10
shuffle2	11,14,9,12

vmovdqa32		(_ZETAS_QINV+272)*4(%rsi),%zmm17
vmovdqa32		(_ZETAS+272)*4(%rsi),%zmm18

butterfly	13,16

vmovdqa32		(_ZETAS_QINV+288)*4(%rsi),%zmm17
vmovdqa32		(_ZETAS+288)*4(%rsi),%zmm18

butterfly	15,28

vmovdqa32		(_ZETAS_QINV+304)*4(%rsi),%zmm17
vmovdqa32		(_ZETAS+304)*4(%rsi),%zmm18

butterfly	22,2

vmovdqa32		(_ZETAS_QINV+320)*4(%rsi),%zmm17
vmovdqa32		(_ZETAS+320)*4(%rsi),%zmm18

butterfly	1,4

vmovdqa32		(_ZETAS_QINV+336)*4(%rsi),%zmm17
vmovdqa32		(_ZETAS+336)*4(%rsi),%zmm18

butterfly	3,6

vmovdqa32		(_ZETAS_QINV+352)*4(%rsi),%zmm17
vmovdqa32		(_ZETAS+352)*4(%rsi),%zmm18

butterfly	5,8

vmovdqa32		(_ZETAS_QINV+368)*4(%rsi),%zmm17
vmovdqa32		(_ZETAS+368)*4(%rsi),%zmm18

butterfly	7,10

vmovdqa32		(_ZETAS_QINV+384)*4(%rsi),%zmm17
vmovdqa32		(_ZETAS+384)*4(%rsi),%zmm18

butterfly	9,12

/* level 7 */
shuffle1	13,16,11,14
shuffle1	15,28,13,16
shuffle1	22,2,15,28
shuffle1	1,4,22,2
shuffle1	3,6,1,4
shuffle1	5,8,3,6
shuffle1	7,10,5,8
shuffle1	9,12,7,10

vmovdqa32		(_ZETAS_QINV+400)*4(%rsi),%zmm17
vmovdqa32		(_ZETAS+400)*4(%rsi),%zmm18
vpsrlq		    $32,%zmm17,%zmm29
vmovshdup	    %zmm18,%zmm30

butterfly	11,14,17,29,18,30

vmovdqa32		(_ZETAS_QINV+416)*4(%rsi),%zmm17
vmovdqa32		(_ZETAS+416)*4(%rsi),%zmm18
vpsrlq		    $32,%zmm17,%zmm29
vmovshdup	    %zmm18,%zmm30

butterfly	13,16,17,29,18,30

vmovdqa32		(_ZETAS_QINV+432)*4(%rsi),%zmm17
vmovdqa32		(_ZETAS+432)*4(%rsi),%zmm18
vpsrlq		    $32,%zmm17,%zmm29
vmovshdup	    %zmm18,%zmm30

butterfly	15,28,17,29,18,30

vmovdqa32		(_ZETAS_QINV+448)*4(%rsi),%zmm17
vmovdqa32		(_ZETAS+448)*4(%rsi),%zmm18
vpsrlq		    $32,%zmm17,%zmm29
vmovshdup	    %zmm18,%zmm30

butterfly	22,2,17,29,18,30

vmovdqa32		(_ZETAS_QINV+464)*4(%rsi),%zmm17
vmovdqa32		(_ZETAS+464)*4(%rsi),%zmm18
vpsrlq		    $32,%zmm17,%zmm29
vmovshdup	    %zmm18,%zmm30

butterfly	1,4,17,29,18,30

vmovdqa32		(_ZETAS_QINV+480)*4(%rsi),%zmm17
vmovdqa32		(_ZETAS+480)*4(%rsi),%zmm18
vpsrlq		    $32,%zmm17,%zmm29
vmovshdup	    %zmm18,%zmm30

butterfly	3,6,17,29,18,30

vmovdqa32		(_ZETAS_QINV+496)*4(%rsi),%zmm17
vmovdqa32		(_ZETAS+496)*4(%rsi),%zmm18
vpsrlq		    $32,%zmm17,%zmm29
vmovshdup	    %zmm18,%zmm30

butterfly	5,8,17,29,18,30

vmovdqa32		(_ZETAS_QINV+512)*4(%rsi),%zmm17
vmovdqa32		(_ZETAS+512)*4(%rsi),%zmm18
vpsrlq		    $32,%zmm17,%zmm29
vmovshdup	    %zmm18,%zmm30

butterfly	7,10,17,29,18,30

//store
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
.endm

.text
.global cdecl(smallntt_avx)
cdecl(smallntt_avx):
mov     $0xAAAA, %eax
kmovw   %eax, %k7
mov     $0x0F0F, %eax
kmovw   %eax, %k6
vmovdqa32		_16XQ*4(%rsi),%zmm0

levels0t7	

ret

