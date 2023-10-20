//preprocessor macro
#if defined(__WIN32__) || defined(__APPLE__)
#define cdecl(s) _##s
#else
#define cdecl(s) s
#endif
.macro pack1 rh0,rh1,rh2,rh3
vpslld       $8,%ymm\rh0,%ymm\rh0
vpor         %ymm\rh0,%ymm\rh1,%ymm\rh0
vpslld       $8,%ymm\rh0,%ymm\rh0
vpor         %ymm\rh0,%ymm\rh2,%ymm\rh0
vpslld       $8,%ymm\rh0,%ymm\rh0
vpor         %ymm\rh0,%ymm\rh3,%ymm\rh0
.endm

.macro pack2 rh0,rh1,rh2
vpslld       $8,%ymm\rh0,%ymm\rh0
vpor         %ymm\rh0,%ymm\rh1,%ymm\rh0
vpslld       $8,%ymm\rh0,%ymm\rh0
vpor         %ymm\rh0,%ymm\rh2,%ymm\rh0
.endm

.macro prepares1 off
# s[0][0] - s[0][7]
vmovdqa      (\off)(%rdx),%ymm0  
# s[1][0] - s[1][7]
vmovdqa      (1024+\off)(%rdx),%ymm1
# s[2][0] - s[2][7]
vmovdqa      (2048+\off)(%rdx),%ymm2
# s[3][0] - s[3][7]
vmovdqa      (3072+\off)(%rdx),%ymm3


#  s[4][0] - s[4][7]
vmovdqa      (4096+\off)(%rdx),%ymm4  
# s[5][0] - s[5][7]
vmovdqa      (5120+\off)(%rdx),%ymm5
# s[6][0] - s[6][7]
vmovdqa      (6144+\off)(%rdx),%ymm6

# add eta s1
vpaddd       %ymm0,%ymm15,%ymm0
vpaddd       %ymm1,%ymm15,%ymm1
vpaddd       %ymm2,%ymm15,%ymm2
vpaddd       %ymm3,%ymm15,%ymm3

vpaddd       %ymm4,%ymm15,%ymm4
vpaddd       %ymm5,%ymm15,%ymm5
vpaddd       %ymm6,%ymm15,%ymm6

# pack s1 for s_table[i+N]
pack1 0,1,2,3
pack2 4,5,6

vpsubd       %ymm0,%ymm14,%ymm8
vpsubd       %ymm4,%ymm13,%ymm9

vmovdqa      %ymm8,(\off)(%rdi)
vmovdqa      %ymm0,(1024+\off)(%rdi)
vmovdqa      %ymm9,(\off)(%rsi)
vmovdqa      %ymm4,(1024+\off)(%rsi)
.endm


.macro prepares2 off
# s[0][0] - s[0][7]
vmovdqa      (\off)(%rdx),%ymm0  
# s[1][0] - s[1][7]
vmovdqa      (1024+\off)(%rdx),%ymm1
# s[2][0] - s[2][7]
vmovdqa      (2048+\off)(%rdx),%ymm2
# s[3][0] - s[3][7]
vmovdqa      (3072+\off)(%rdx),%ymm3

#  s[4][0] - s[4][7]
vmovdqa      (4096+\off)(%rdx),%ymm4  
# s[5][0] - s[5][7]
vmovdqa      (5120+\off)(%rdx),%ymm5
# s[6][0] - s[6][7]
vmovdqa      (6144+\off)(%rdx),%ymm6
# s[7][0] - s[7][7]
vmovdqa      (7168+\off)(%rdx),%ymm7

# add eta s1
vpaddd       %ymm0,%ymm15,%ymm0
vpaddd       %ymm1,%ymm15,%ymm1
vpaddd       %ymm2,%ymm15,%ymm2
vpaddd       %ymm3,%ymm15,%ymm3

vpaddd       %ymm4,%ymm15,%ymm4
vpaddd       %ymm5,%ymm15,%ymm5
vpaddd       %ymm6,%ymm15,%ymm6
vpaddd       %ymm7,%ymm15,%ymm7

# pack s1 for s_table[i+N]
pack1 0,1,2,3
pack1 4,5,6,7

vpsubd       %ymm0,%ymm14,%ymm8
vpsubd       %ymm4,%ymm14,%ymm9

vmovdqa      %ymm8,(\off)(%rdi)
vmovdqa      %ymm0,(1024+\off)(%rdi)
vmovdqa      %ymm9,(\off)(%rsi)
vmovdqa      %ymm4,(1024+\off)(%rsi)
.endm


.p2align 5   //instructs the assembler to align the following instruction or data on a boundary that is a power of 2 and here is equal to 2^5, or 32 bytes
.global cdecl(prepare_s1_table_avx2)
cdecl(prepare_s1_table_avx2):

vpbroadcastd		_8xeta(%rip),%ymm15
vpbroadcastd		_8xmask1(%rip),%ymm14
vpbroadcastd		_8xmask2(%rip),%ymm13


prepares1  0
prepares1  32
prepares1  64
prepares1  96
prepares1  128
prepares1  160
prepares1  192
prepares1  224
prepares1  256
prepares1  288
prepares1  320
prepares1  352
prepares1  384
prepares1  416
prepares1  448
prepares1  480
prepares1  512
prepares1  544
prepares1  576
prepares1  608
prepares1  640
prepares1  672
prepares1  704
prepares1  736
prepares1  768
prepares1  800
prepares1  832
prepares1  864
prepares1  896
prepares1  928
prepares1  960
prepares1  992
ret






.p2align 5   //instructs the assembler to align the following instruction or data on a boundary that is a power of 2 and here is equal to 2^5, or 32 bytes
.global cdecl(prepare_s2_table_avx2)
cdecl(prepare_s2_table_avx2):
vpbroadcastd		_8xeta(%rip),%ymm15
vpbroadcastd		_8xmask1(%rip),%ymm14



prepares2  0
prepares2  32
prepares2  64
prepares2  96
prepares2  128
prepares2  160
prepares2  192
prepares2  224
prepares2  256
prepares2  288
prepares2  320
prepares2  352
prepares2  384
prepares2  416
prepares2  448
prepares2  480
prepares2  512
prepares2  544
prepares2  576
prepares2  608
prepares2  640
prepares2  672
prepares2  704
prepares2  736
prepares2  768
prepares2  800
prepares2  832
prepares2  864
prepares2  896
prepares2  928
prepares2  960
prepares2  992

ret