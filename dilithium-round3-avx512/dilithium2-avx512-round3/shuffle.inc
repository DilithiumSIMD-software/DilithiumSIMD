.macro shuffle8 r0,r1,r2,r3
vshufi32x4	$0x44,%zmm\r1,%zmm\r0,%zmm\r2
vshufi32x4	$0xEE,%zmm\r1,%zmm\r0,%zmm\r3
.endm

.macro shuffle4 r0,r1,r2,r3
vpermq	$0x4E,%zmm\r1,%zmm\r2
vpblendmd %zmm\r0,%zmm\r2,%zmm\r2{%k6}
vpermq	$0x4E,%zmm\r0,%zmm\r3
vpblendmd %zmm\r3,%zmm\r1,%zmm\r3{%k6}
.endm

.macro shuffle2 r0,r1,r2,r3
vpunpcklqdq	%zmm\r1,%zmm\r0,%zmm\r2
vpunpckhqdq	%zmm\r1,%zmm\r0,%zmm\r3
.endm

.macro shuffle1 r0,r1,r2,r3
vpsllq      $32,%zmm\r1,%zmm31
vpblendmd   %zmm31,%zmm\r0,%zmm\r2{%k7}
vmovshdup   %zmm\r0,%zmm\r0
vpblendmd   %zmm\r1,%zmm\r0,%zmm\r3{%k7}
.endm
