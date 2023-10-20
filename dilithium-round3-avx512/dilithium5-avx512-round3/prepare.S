//preprocessor macro
#if defined(__WIN32__) || defined(__APPLE__)
#define cdecl(s) _##s
#else
#define cdecl(s) s
#endif

.macro pack rh0,rh1,rh2,rh3
vpslld        $8,%zmm\rh0,%zmm\rh0
vpord         %zmm\rh0,%zmm\rh1,%zmm\rh0
vpslld        $8,%zmm\rh0,%zmm\rh0
vpord         %zmm\rh0,%zmm\rh2,%zmm\rh0
vpslld        $8,%zmm\rh0,%zmm\rh0
vpord         %zmm\rh0,%zmm\rh3,%zmm\rh0
.endm


.macro prepare off
//load coeffes off=0:index 0-15    off=448:index 112-127
vmovdqu32      (\off)(%rsi),%zmm0  
vmovdqu32      (1024+\off)(%rsi),%zmm1
vmovdqu32      (2048+\off)(%rsi),%zmm2
vmovdqu32      (3072+\off)(%rsi),%zmm3
//load coeffes off=0:index 16-31    off=448:index 128-143
vmovdqu32      (64+\off)(%rsi),%zmm4  
vmovdqu32      (64+1024+\off)(%rsi),%zmm5
vmovdqu32      (64+2048+\off)(%rsi),%zmm6
vmovdqu32      (64+3072+\off)(%rsi),%zmm7
//load coeffes off=0:index 32-47    off=448:index  144-159
vmovdqu32      (128+\off)(%rsi),%zmm8
vmovdqu32      (128+1024+\off)(%rsi),%zmm9
vmovdqu32      (128+2048+\off)(%rsi),%zmm10
vmovdqu32      (128+3072+\off)(%rsi),%zmm11
//load coeffes off=0:index 48-63    off=448:index 160-175
vmovdqu32      (192+\off)(%rsi),%zmm12
vmovdqu32      (192+1024+\off)(%rsi),%zmm13
vmovdqu32      (192+2048+\off)(%rsi),%zmm14
vmovdqu32      (192+3072+\off)(%rsi),%zmm15 
//load coeffes off=0:index 64-79    off=448:index 176-191
vmovdqu32      (256+\off)(%rsi),%zmm16
vmovdqu32      (256+1024+\off)(%rsi),%zmm17
vmovdqu32      (256+2048+\off)(%rsi),%zmm18
vmovdqu32      (256+3072+\off)(%rsi),%zmm19
//load coeffes off=0:index 80-95    off=448:index 192-207
vmovdqu32      (320+\off)(%rsi),%zmm20
vmovdqu32      (320+1024+\off)(%rsi),%zmm21
vmovdqu32      (320+2048+\off)(%rsi),%zmm22
vmovdqu32      (320+3072+\off)(%rsi),%zmm23
//load coeffes off=0:index 96-111    off=448:index 208-223
vmovdqu32      (384+\off)(%rsi),%zmm24
vmovdqu32      (384+1024+\off)(%rsi),%zmm25
vmovdqu32      (384+2048+\off)(%rsi),%zmm26
vmovdqu32      (384+3072+\off)(%rsi),%zmm27

# add eta 
vpaddd       %zmm0,%zmm31,%zmm0
vpaddd       %zmm1,%zmm31,%zmm1
vpaddd       %zmm2,%zmm31,%zmm2
vpaddd       %zmm3,%zmm31,%zmm3

vpaddd       %zmm4,%zmm31,%zmm4
vpaddd       %zmm5,%zmm31,%zmm5
vpaddd       %zmm6,%zmm31,%zmm6
vpaddd       %zmm7,%zmm31,%zmm7

vpaddd       %zmm8,%zmm31,%zmm8
vpaddd       %zmm9,%zmm31,%zmm9
vpaddd       %zmm10,%zmm31,%zmm10
vpaddd       %zmm11,%zmm31,%zmm11

vpaddd       %zmm12,%zmm31,%zmm12
vpaddd       %zmm13,%zmm31,%zmm13
vpaddd       %zmm14,%zmm31,%zmm14
vpaddd       %zmm15,%zmm31,%zmm15

vpaddd       %zmm16,%zmm31,%zmm16
vpaddd       %zmm17,%zmm31,%zmm17
vpaddd       %zmm18,%zmm31,%zmm18
vpaddd       %zmm19,%zmm31,%zmm19

vpaddd       %zmm20,%zmm31,%zmm20
vpaddd       %zmm21,%zmm31,%zmm21
vpaddd       %zmm22,%zmm31,%zmm22
vpaddd       %zmm23,%zmm31,%zmm23

vpaddd       %zmm24,%zmm31,%zmm24
vpaddd       %zmm25,%zmm31,%zmm25
vpaddd       %zmm26,%zmm31,%zmm26
vpaddd       %zmm27,%zmm31,%zmm27

# pack s for s_table[i+N]
pack 0,1,2,3
pack 4,5,6,7
pack 8,9,10,11
pack 12,13,14,15
pack 16,17,18,19
pack 20,21,22,23
pack 24,25,26,27

//compute s_table[i]=mask-s_table[i+N]
vpsubd       %zmm0,%zmm30,%zmm1
vpsubd       %zmm4,%zmm30,%zmm5
vpsubd       %zmm8,%zmm30,%zmm9
vpsubd       %zmm12,%zmm30,%zmm13
vpsubd       %zmm16,%zmm30,%zmm17
vpsubd       %zmm20,%zmm30,%zmm21
vpsubd       %zmm24,%zmm30,%zmm25

//store
vmovdqu32      %zmm1,(\off)(%rdi)
vmovdqu32      %zmm0,(1024+\off)(%rdi)
vmovdqu32      %zmm5,(64+\off)(%rdi)
vmovdqu32      %zmm4,(64+1024+\off)(%rdi)
vmovdqu32      %zmm9,(128+\off)(%rdi)
vmovdqu32      %zmm8,(128+1024+\off)(%rdi)
vmovdqu32      %zmm13,(192+\off)(%rdi)
vmovdqu32      %zmm12,(192+1024+\off)(%rdi)
vmovdqu32      %zmm17,(256+\off)(%rdi)
vmovdqu32      %zmm16,(256+1024+\off)(%rdi)
vmovdqu32      %zmm21,(320+\off)(%rdi)
vmovdqu32      %zmm20,(320+1024+\off)(%rdi)
vmovdqu32      %zmm25,(384+\off)(%rdi)
vmovdqu32      %zmm24,(384+1024+\off)(%rdi)

.endm



.p2align 5   //instructs the assembler to align the following instruction or data on a boundary that is a power of 2 and here is equal to 2^5, or 32 bytes
.global cdecl(prepare_s_table4x)
cdecl(prepare_s_table4x):

vpbroadcastd		_16xeta(%rip),%zmm31
vpbroadcastd		_16xmasks(%rip),%zmm30

prepare 0
prepare 448

//load coeffs index:224-239
vmovdqu32      (896)(%rsi),%zmm0  
vmovdqu32      (896+1024)(%rsi),%zmm1
vmovdqu32      (896+2048)(%rsi),%zmm2
vmovdqu32      (896+3072)(%rsi),%zmm3
//load coeffs index:240-255
vmovdqu32      (960)(%rsi),%zmm4  
vmovdqu32      (960+1024)(%rsi),%zmm5
vmovdqu32      (960+2048)(%rsi),%zmm6
vmovdqu32      (960+3072)(%rsi),%zmm7
# add eta 
vpaddd       %zmm0,%zmm31,%zmm0
vpaddd       %zmm1,%zmm31,%zmm1
vpaddd       %zmm2,%zmm31,%zmm2
vpaddd       %zmm3,%zmm31,%zmm3

vpaddd       %zmm4,%zmm31,%zmm4
vpaddd       %zmm5,%zmm31,%zmm5
vpaddd       %zmm6,%zmm31,%zmm6
vpaddd       %zmm7,%zmm31,%zmm7

# pack s for s_table[i+N]
pack 0,1,2,3
pack 4,5,6,7

//compute s_table[i]=mask-s_table[i+N]
vpsubd       %zmm0,%zmm30,%zmm1
vpsubd       %zmm4,%zmm30,%zmm5

//store
vmovdqu32      %zmm1,(896)(%rdi)
vmovdqu32      %zmm0,(896+1024)(%rdi)
vmovdqu32      %zmm5,(960)(%rdi)
vmovdqu32      %zmm4,(960+1024)(%rdi)

ret


.macro pack3x rh0,rh1,rh2
vpslld        $8,%zmm\rh0,%zmm\rh0
vpord         %zmm\rh0,%zmm\rh1,%zmm\rh0
vpslld        $8,%zmm\rh0,%zmm\rh0
vpord         %zmm\rh0,%zmm\rh2,%zmm\rh0
.endm


.macro prepare3x off
//load coeffes off=0:index 0-15    off=512:index 128-143
vmovdqu32      (\off)(%rsi),%zmm0  
vmovdqu32      (1024+\off)(%rsi),%zmm1
vmovdqu32      (2048+\off)(%rsi),%zmm2
//load coeffes off=0:index 16-31    off=512:index 144-159
vmovdqu32      (64+\off)(%rsi),%zmm3
vmovdqu32      (64+1024+\off)(%rsi),%zmm4
vmovdqu32      (64+2048+\off)(%rsi),%zmm5
//load coeffes off=0:index 32-47    off=512:index  160-175
vmovdqu32      (128+\off)(%rsi),%zmm6
vmovdqu32      (128+1024+\off)(%rsi),%zmm7
vmovdqu32      (128+2048+\off)(%rsi),%zmm8
//load coeffes off=0:index 48-63    off=512:index 176-191
vmovdqu32      (192+\off)(%rsi),%zmm9
vmovdqu32      (192+1024+\off)(%rsi),%zmm10
vmovdqu32      (192+2048+\off)(%rsi),%zmm11
//load coeffes off=0:index 64-79    off=512:index 192-207
vmovdqu32      (256+\off)(%rsi),%zmm12
vmovdqu32      (256+1024+\off)(%rsi),%zmm13
vmovdqu32      (256+2048+\off)(%rsi),%zmm14
//load coeffes off=0:index 80-95    off=512:index 208-223
vmovdqu32      (320+\off)(%rsi),%zmm15
vmovdqu32      (320+1024+\off)(%rsi),%zmm16
vmovdqu32      (320+2048+\off)(%rsi),%zmm17
//load coeffes off=0:index 96-111    off=512:index 224-239
vmovdqu32      (384+\off)(%rsi),%zmm18
vmovdqu32      (384+1024+\off)(%rsi),%zmm19
vmovdqu32      (384+2048+\off)(%rsi),%zmm20
//load coeffes off=0:index 112-127   off=512:index 240-255
vmovdqu32      (448+\off)(%rsi),%zmm21
vmovdqu32      (448+1024+\off)(%rsi),%zmm22
vmovdqu32      (448+2048+\off)(%rsi),%zmm23

# add eta 
vpaddd       %zmm0,%zmm31,%zmm0
vpaddd       %zmm1,%zmm31,%zmm1
vpaddd       %zmm2,%zmm31,%zmm2

vpaddd       %zmm3,%zmm31,%zmm3
vpaddd       %zmm4,%zmm31,%zmm4
vpaddd       %zmm5,%zmm31,%zmm5

vpaddd       %zmm6,%zmm31,%zmm6
vpaddd       %zmm7,%zmm31,%zmm7
vpaddd       %zmm8,%zmm31,%zmm8

vpaddd       %zmm9,%zmm31,%zmm9
vpaddd       %zmm10,%zmm31,%zmm10
vpaddd       %zmm11,%zmm31,%zmm11

vpaddd       %zmm12,%zmm31,%zmm12
vpaddd       %zmm13,%zmm31,%zmm13
vpaddd       %zmm14,%zmm31,%zmm14

vpaddd       %zmm15,%zmm31,%zmm15
vpaddd       %zmm16,%zmm31,%zmm16
vpaddd       %zmm17,%zmm31,%zmm17

vpaddd       %zmm18,%zmm31,%zmm18
vpaddd       %zmm19,%zmm31,%zmm19
vpaddd       %zmm20,%zmm31,%zmm20

vpaddd       %zmm21,%zmm31,%zmm21
vpaddd       %zmm22,%zmm31,%zmm22
vpaddd       %zmm23,%zmm31,%zmm23


# pack s for s_table[i+N]
pack3x 0,1,2
pack3x 3,4,5
pack3x 6,7,8
pack3x 9,10,11
pack3x 12,13,14
pack3x 15,16,17
pack3x 18,19,20
pack3x 21,22,23

//compute s_table[i]=mask-s_table[i+N]
vpsubd       %zmm0,%zmm30,%zmm1
vpsubd       %zmm3,%zmm30,%zmm4
vpsubd       %zmm6,%zmm30,%zmm7
vpsubd       %zmm9,%zmm30,%zmm10
vpsubd       %zmm12,%zmm30,%zmm13
vpsubd       %zmm15,%zmm30,%zmm16
vpsubd       %zmm18,%zmm30,%zmm19
vpsubd       %zmm21,%zmm30,%zmm22

//store
vmovdqu32      %zmm1,(\off)(%rdi)
vmovdqu32      %zmm0,(1024+\off)(%rdi)
vmovdqu32      %zmm4,(64+\off)(%rdi)
vmovdqu32      %zmm3,(64+1024+\off)(%rdi)
vmovdqu32      %zmm7,(128+\off)(%rdi)
vmovdqu32      %zmm6,(128+1024+\off)(%rdi)
vmovdqu32      %zmm10,(192+\off)(%rdi)
vmovdqu32      %zmm9,(192+1024+\off)(%rdi)
vmovdqu32      %zmm13,(256+\off)(%rdi)
vmovdqu32      %zmm12,(256+1024+\off)(%rdi)
vmovdqu32      %zmm16,(320+\off)(%rdi)
vmovdqu32      %zmm15,(320+1024+\off)(%rdi)
vmovdqu32      %zmm19,(384+\off)(%rdi)
vmovdqu32      %zmm18,(384+1024+\off)(%rdi)
vmovdqu32      %zmm22,(448+\off)(%rdi)
vmovdqu32      %zmm21,(448+1024+\off)(%rdi)
.endm


.p2align 5   //instructs the assembler to align the following instruction or data on a boundary that is a power of 2 and here is equal to 2^5, or 32 bytes
.global cdecl(prepare_s_table3x)
cdecl(prepare_s_table3x):

vpbroadcastd		_16xeta(%rip),%zmm31
vpbroadcastd		_16xmask2(%rip),%zmm30

prepare3x 0
prepare3x 512


ret