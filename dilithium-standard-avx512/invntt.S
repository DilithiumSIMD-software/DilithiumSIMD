#include "consts.h"
.include "shuffle.inc"

.macro butterfly l,h,zl0=17,zl1=17,zh0=18,zh1=18
vpsubd		%zmm\h,%zmm\l,%zmm20
vpaddd		%zmm\h,%zmm\l,%zmm\l
vmovshdup	%zmm20,%zmm\h

vpmuldq		%zmm\zl0,%zmm20,%zmm21
vpmuldq		%zmm\zl1,%zmm\h,%zmm22
vpmuldq		%zmm0,%zmm21,%zmm21
vpmuldq		%zmm0,%zmm22,%zmm22

vpmuldq		%zmm\zh0,%zmm20,%zmm20
vpmuldq		%zmm\zh1,%zmm\h,%zmm\h

vpsubq		%zmm21,%zmm20,%zmm20
vpsubq		%zmm22,%zmm\h,%zmm\h
vmovshdup	%zmm20,%zmm20
vpblendmd	%zmm\h,%zmm20,%zmm\h{%k7}
.endm

.macro montmul rl,rh,zl0=17,zl1=17,zh0=18,zh1=18
vmovshdup	%zmm\rl,%zmm\rh
vpmuldq		%zmm\zl0,%zmm\rl,%zmm21
vpmuldq		%zmm\zl1,%zmm\rh,%zmm22
vpmuldq		%zmm0,%zmm21,%zmm21
vpmuldq		%zmm0,%zmm22,%zmm22

vpmuldq		%zmm\zh0,%zmm\rl,%zmm\rl
vpmuldq		%zmm\zh1,%zmm\rh,%zmm\rh

vpsubq		%zmm21,%zmm\rl,%zmm\rl
vpsubq		%zmm22,%zmm\rh,%zmm\rh
vmovshdup	%zmm\rl,%zmm\rl
vpblendmd	%zmm\rh,%zmm\rl,%zmm\rh{%k7}
.endm

.macro levels0t7 
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

/* level 7 */
vmovdqa32		(_ZETASINV_QINV+0)*4(%rsi),%zmm17
vmovdqa32		(_ZETASINV+0)*4(%rsi),%zmm18
vpsrlq		    $32,%zmm17,%zmm29
vmovshdup	    %zmm18,%zmm30

butterfly	1,2,17,29,18,30

vmovdqa32		(_ZETASINV_QINV+16)*4(%rsi),%zmm17
vmovdqa32		(_ZETASINV+16)*4(%rsi),%zmm18
vpsrlq		    $32,%zmm17,%zmm29
vmovshdup	    %zmm18,%zmm30

butterfly	3,4,17,29,18,30

vmovdqa32		(_ZETASINV_QINV+32)*4(%rsi),%zmm17
vmovdqa32		(_ZETASINV+32)*4(%rsi),%zmm18
vpsrlq		    $32,%zmm17,%zmm29
vmovshdup	    %zmm18,%zmm30

butterfly	5,6,17,29,18,30

vmovdqa32		(_ZETASINV_QINV+48)*4(%rsi),%zmm17
vmovdqa32		(_ZETASINV+48)*4(%rsi),%zmm18
vpsrlq		    $32,%zmm17,%zmm29
vmovshdup	    %zmm18,%zmm30

butterfly	7,8,17,29,18,30

vmovdqa32		(_ZETASINV_QINV+64)*4(%rsi),%zmm17
vmovdqa32		(_ZETASINV+64)*4(%rsi),%zmm18
vpsrlq		    $32,%zmm17,%zmm29
vmovshdup	    %zmm18,%zmm30

butterfly	9,10,17,29,18,30

vmovdqa32		(_ZETASINV_QINV+80)*4(%rsi),%zmm17
vmovdqa32		(_ZETASINV+80)*4(%rsi),%zmm18
vpsrlq		    $32,%zmm17,%zmm29
vmovshdup	    %zmm18,%zmm30

butterfly	11,12,17,29,18,30

vmovdqa32		(_ZETASINV_QINV+96)*4(%rsi),%zmm17
vmovdqa32		(_ZETASINV+96)*4(%rsi),%zmm18
vpsrlq		    $32,%zmm17,%zmm29
vmovshdup	    %zmm18,%zmm30

butterfly	13,14,17,29,18,30

vmovdqa32		(_ZETASINV_QINV+112)*4(%rsi),%zmm17
vmovdqa32		(_ZETASINV+112)*4(%rsi),%zmm18
vpsrlq		    $32,%zmm17,%zmm29
vmovshdup	    %zmm18,%zmm30

butterfly	15,16,17,29,18,30


shuffle1	1,2,27,28   
shuffle1	3,4,1,2    
shuffle1	5,6,3,4    
shuffle1	7,8,5,6    
shuffle1	9,10,7,8  
shuffle1	11,12,9,10 
shuffle1	13,14,11,12
shuffle1	15,16,13,14

vmovdqa32		(_ZETASINV_QINV+128)*4(%rsi),%zmm17
vmovdqa32		(_ZETASINV+128)*4(%rsi),%zmm18

butterfly	27,28 

vmovdqa32		(_ZETASINV_QINV+144)*4(%rsi),%zmm17
vmovdqa32		(_ZETASINV+144)*4(%rsi),%zmm18

butterfly	1,2

vmovdqa32		(_ZETASINV_QINV+160)*4(%rsi),%zmm17
vmovdqa32		(_ZETASINV+160)*4(%rsi),%zmm18

butterfly	3,4

vmovdqa32		(_ZETASINV_QINV+176)*4(%rsi),%zmm17
vmovdqa32		(_ZETASINV+176)*4(%rsi),%zmm18

butterfly	5,6

vmovdqa32		(_ZETASINV_QINV+192)*4(%rsi),%zmm17
vmovdqa32		(_ZETASINV+192)*4(%rsi),%zmm18

butterfly	7,8

vmovdqa32		(_ZETASINV_QINV+208)*4(%rsi),%zmm17
vmovdqa32		(_ZETASINV+208)*4(%rsi),%zmm18

butterfly	9,10

vmovdqa32		(_ZETASINV_QINV+224)*4(%rsi),%zmm17
vmovdqa32		(_ZETASINV+224)*4(%rsi),%zmm18

butterfly	11,12

vmovdqa32		(_ZETASINV_QINV+240)*4(%rsi),%zmm17
vmovdqa32		(_ZETASINV+240)*4(%rsi),%zmm18

butterfly	13,14

shuffle2	27,28,15,16       
shuffle2	1,2,27,28         
shuffle2	3,4,1,2           
shuffle2	5,6,3,4           
shuffle2	7,8,5,6      
shuffle2	9,10,7,8    
shuffle2	11,12,9,10 
shuffle2	13,14,11,12

/* level 5 */	
vmovdqa32		(_ZETASINV_QINV+256)*4(%rsi),%zmm17
vmovdqa32		(_ZETASINV+256)*4(%rsi),%zmm18

butterfly	15,16

vmovdqa32		(_ZETASINV_QINV+272)*4(%rsi),%zmm17
vmovdqa32		(_ZETASINV+272)*4(%rsi),%zmm18

butterfly	27,28 

vmovdqa32		(_ZETASINV_QINV+288)*4(%rsi),%zmm17
vmovdqa32		(_ZETASINV+288)*4(%rsi),%zmm18

butterfly	1,2 

vmovdqa32		(_ZETASINV_QINV+304)*4(%rsi),%zmm17
vmovdqa32		(_ZETASINV+304)*4(%rsi),%zmm18

butterfly	3,4

vmovdqa32		(_ZETASINV_QINV+320)*4(%rsi),%zmm17
vmovdqa32		(_ZETASINV+320)*4(%rsi),%zmm18

butterfly	5,6 

vmovdqa32		(_ZETASINV_QINV+336)*4(%rsi),%zmm17
vmovdqa32		(_ZETASINV+336)*4(%rsi),%zmm18

butterfly	7,8

vmovdqa32		(_ZETASINV_QINV+352)*4(%rsi),%zmm17
vmovdqa32		(_ZETASINV+352)*4(%rsi),%zmm18

butterfly	9,10

vmovdqa32		(_ZETASINV_QINV+368)*4(%rsi),%zmm17
vmovdqa32		(_ZETASINV+368)*4(%rsi),%zmm18

butterfly	11,12

shuffle4    15,16,13,14 	 
shuffle4    27,28,15,16 	
shuffle4    1,2,27,28  	
shuffle4    3,4,1,2   	
shuffle4    5,6,3,4 	
shuffle4    7,8,5,6 	
shuffle4    9,10,7,8 	
shuffle4    11,12,9,10

/* level 4 */
vmovdqa32		(_ZETASINV_QINV+384)*4(%rsi),%zmm17
vmovdqa32		(_ZETASINV+384)*4(%rsi),%zmm18

butterfly	13,14

vmovdqa32		(_ZETASINV_QINV+400)*4(%rsi),%zmm17
vmovdqa32		(_ZETASINV+400)*4(%rsi),%zmm18

butterfly	15,16

vmovdqa32		(_ZETASINV_QINV+416)*4(%rsi),%zmm17
vmovdqa32		(_ZETASINV+416)*4(%rsi),%zmm18

butterfly	27,28

vmovdqa32		(_ZETASINV_QINV+432)*4(%rsi),%zmm17
vmovdqa32		(_ZETASINV+432)*4(%rsi),%zmm18

butterfly	1,2

vmovdqa32		(_ZETASINV_QINV+448)*4(%rsi),%zmm17
vmovdqa32		(_ZETASINV+448)*4(%rsi),%zmm18

butterfly	3,4

vmovdqa32		(_ZETASINV_QINV+464)*4(%rsi),%zmm17
vmovdqa32		(_ZETASINV+464)*4(%rsi),%zmm18

butterfly	5,6

vmovdqa32		(_ZETASINV_QINV+480)*4(%rsi),%zmm17
vmovdqa32		(_ZETASINV+480)*4(%rsi),%zmm18

butterfly	7,8

vmovdqa32		(_ZETASINV_QINV+496)*4(%rsi),%zmm17
vmovdqa32		(_ZETASINV+496)*4(%rsi),%zmm18

butterfly	9,10

shuffle8	13,14,11,12
shuffle8	15,16,13,14
shuffle8	27,28,15,16
shuffle8	1,2,27,28
shuffle8	3,4,1,2
shuffle8	5,6,3,4
shuffle8	7,8,5,6
shuffle8	9,10,7,8

/* level 3 */
vpbroadcastd		(_ZETASINV_QINV+512)*4(%rsi),%zmm17
vpbroadcastd		(_ZETASINV+512)*4(%rsi),%zmm18

butterfly	11,12

vpbroadcastd		(_ZETASINV_QINV+513)*4(%rsi),%zmm17
vpbroadcastd		(_ZETASINV+513)*4(%rsi),%zmm18

butterfly	13,14

vpbroadcastd		(_ZETASINV_QINV+514)*4(%rsi),%zmm17
vpbroadcastd		(_ZETASINV+514)*4(%rsi),%zmm18

butterfly	15,16

vpbroadcastd		(_ZETASINV_QINV+515)*4(%rsi),%zmm17
vpbroadcastd		(_ZETASINV+515)*4(%rsi),%zmm18

butterfly	27,28

vpbroadcastd		(_ZETASINV_QINV+516)*4(%rsi),%zmm17
vpbroadcastd		(_ZETASINV+516)*4(%rsi),%zmm18

butterfly	1,2

vpbroadcastd		(_ZETASINV_QINV+517)*4(%rsi),%zmm17
vpbroadcastd		(_ZETASINV+517)*4(%rsi),%zmm18

butterfly	3,4

vpbroadcastd		(_ZETASINV_QINV+518)*4(%rsi),%zmm17
vpbroadcastd		(_ZETASINV+518)*4(%rsi),%zmm18

butterfly	5,6

vpbroadcastd		(_ZETASINV_QINV+519)*4(%rsi),%zmm17
vpbroadcastd		(_ZETASINV+519)*4(%rsi),%zmm18

butterfly	7,8

/* level 2*/
vpbroadcastd		(_ZETASINV_QINV+520)*4(%rsi),%zmm17
vpbroadcastd		(_ZETASINV+520)*4(%rsi),%zmm18

butterfly	11,13
butterfly	12,14

vpbroadcastd		(_ZETASINV_QINV+521)*4(%rsi),%zmm17
vpbroadcastd		(_ZETASINV+521)*4(%rsi),%zmm18
butterfly	15,27
butterfly	16,28

vpbroadcastd		(_ZETASINV_QINV+522)*4(%rsi),%zmm17
vpbroadcastd		(_ZETASINV+522)*4(%rsi),%zmm18
butterfly	1,3
butterfly	2,4

vpbroadcastd		(_ZETASINV_QINV+523)*4(%rsi),%zmm17
vpbroadcastd		(_ZETASINV+523)*4(%rsi),%zmm18
butterfly	5,7
butterfly	6,8

/* level 1*/
// 2 zetas
vpbroadcastd		(_ZETASINV_QINV+524)*4(%rsi),%zmm17
vpbroadcastd		(_ZETASINV+524)*4(%rsi),%zmm18
butterfly	11,15
butterfly	12,16
butterfly	13,27
butterfly	14,28

vpbroadcastd		(_ZETASINV_QINV+525)*4(%rsi),%zmm17
vpbroadcastd		(_ZETASINV+525)*4(%rsi),%zmm18
butterfly	1,5
butterfly	2,6
butterfly	3,7
butterfly	4,8
/* level 0*/
// 1 zetas
vpbroadcastd		(_ZETASINV_QINV+526)*4(%rsi),%zmm17
vpbroadcastd		(_ZETASINV+526)*4(%rsi),%zmm18

butterfly	11,1
butterfly	12,2
butterfly	13,3
butterfly	14,4
butterfly	15,5
butterfly	16,6
butterfly	27,7
butterfly	28,8

vpbroadcastd		(_ZETASINV_QINV+527)*4(%rsi),%zmm17
vpbroadcastd		(_ZETASINV+527)*4(%rsi),%zmm18

montmul  11,9
montmul  12,11
montmul  13,12
montmul  14,13
montmul  15,14
montmul  16,15
montmul  27,16
montmul  28,27

vmovdqa32		%zmm9,  0(%rdi)
vmovdqa32		%zmm11, 64(%rdi)
vmovdqa32		%zmm12,128(%rdi)
vmovdqa32	 	%zmm13,192(%rdi)
vmovdqa32		%zmm14,256(%rdi)
vmovdqa32		%zmm15,320(%rdi)
vmovdqa32		%zmm16,384(%rdi)
vmovdqa32	 	%zmm27,448(%rdi)
// load a[128]~a[255]
vmovdqa32		%zmm1,512(%rdi)
vmovdqa32		%zmm2,576(%rdi)
vmovdqa32		%zmm3,640(%rdi)
vmovdqa32	 	%zmm4,704(%rdi)
vmovdqa32		%zmm5,768(%rdi)
vmovdqa32		%zmm6,832(%rdi)
vmovdqa32		%zmm7,896(%rdi)
vmovdqa32	 	%zmm8,960(%rdi)
.endm

.text
.global cdecl(invntt_avx)
cdecl(invntt_avx):
mov     $0xAAAA, %eax
kmovw   %eax, %k7
mov     $0x0F0F, %eax
kmovw   %eax, %k6
vmovdqa32		_16XQ*4(%rsi),%zmm0

levels0t7	
ret

