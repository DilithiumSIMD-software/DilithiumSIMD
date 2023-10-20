//preprocessor macro
#if defined(__WIN32__) || defined(__APPLE__)
#define cdecl(s) _##s
#else
#define cdecl(s) s
#endif

.macro pack rh0,rh1,rh2,rh3
vpslld       $8,%ymm\rh0,%ymm\rh0
vpor         %ymm\rh0,%ymm\rh1,%ymm\rh0
vpslld       $8,%ymm\rh0,%ymm\rh0
vpor         %ymm\rh0,%ymm\rh2,%ymm\rh0
vpslld       $8,%ymm\rh0,%ymm\rh0
vpor         %ymm\rh0,%ymm\rh3,%ymm\rh0
.endm

.macro prepare off
vmovdqa      (\off)(%rsi),%ymm0  
vmovdqa      (1024+\off)(%rsi),%ymm1
vmovdqa      (2048+\off)(%rsi),%ymm2
vmovdqa      (3072+\off)(%rsi),%ymm3
vmovdqa      (32+\off)(%rsi),%ymm4  
vmovdqa      (32+1024+\off)(%rsi),%ymm5
vmovdqa      (32+2048+\off)(%rsi),%ymm6
vmovdqa      (32+3072+\off)(%rsi),%ymm7

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
pack 0,1,2,3
pack 4,5,6,7

vpsubd       %ymm0,%ymm14,%ymm8
vpsubd       %ymm4,%ymm14,%ymm9

vmovdqa      %ymm8,(\off)(%rdi)
vmovdqa      %ymm0,(1024+\off)(%rdi)
vmovdqa      %ymm9,(32+\off)(%rdi)
vmovdqa      %ymm4,(32+1024+\off)(%rdi)
.endm

.p2align 5   //instructs the assembler to align the following instruction or data on a boundary that is a power of 2 and here is equal to 2^5, or 32 bytes
.global cdecl(prepare_s1_table)
cdecl(prepare_s1_table):

vpbroadcastd		_8xeta(%rip),%ymm15
vpbroadcastd		_8xmasks(%rip),%ymm14

prepare 0
prepare 64
prepare 128
prepare 192
prepare 256
prepare 320
prepare 384
prepare 448
prepare 512
prepare 576
prepare 640
prepare 704
prepare 768
prepare 832
prepare 896
prepare 960

ret


.p2align 5   //instructs the assembler to align the following instruction or data on a boundary that is a power of 2 and here is equal to 2^5, or 32 bytes
.global cdecl(prepare_s2_table)
cdecl(prepare_s2_table):

vpbroadcastd		_8xeta(%rip),%ymm15
vpbroadcastd		_8xmasks(%rip),%ymm14

prepare 0
prepare 64
prepare 128
prepare 192
prepare 256
prepare 320
prepare 384
prepare 448
prepare 512
prepare 576
prepare 640
prepare 704
prepare 768
prepare 832
prepare 896
prepare 960

ret

