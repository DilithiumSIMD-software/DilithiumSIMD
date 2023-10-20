//preprocessor macro
#if defined(__WIN32__) || defined(__APPLE__)
#define cdecl(s) _##s
#else
#define cdecl(s) s
#endif


.macro accanswer off
vmovdqu   (\off)(%rsi),%ymm0
vmovdqu   (32+\off)(%rsi),%ymm1
vmovdqu   (64+\off)(%rsi),%ymm2
vmovdqu   (96+\off)(%rsi),%ymm3
vmovdqu   (128+\off)(%rsi),%ymm4
vmovdqu   (160+\off)(%rsi),%ymm5
vmovdqu   (192+\off)(%rsi),%ymm6
vmovdqu   (224+\off)(%rsi),%ymm7
vmovdqu   (\off)(%rdi),%ymm8
vmovdqu   (32+\off)(%rdi),%ymm9
vmovdqu   (64+\off)(%rdi),%ymm10
vmovdqu   (96+\off)(%rdi),%ymm11

vpaddd    %ymm0,%ymm8,%ymm0
vpaddd    %ymm1,%ymm9,%ymm1
vpaddd    %ymm2,%ymm10,%ymm2
vpaddd    %ymm3,%ymm11,%ymm3

vmovdqu   (128+\off)(%rdi),%ymm8
vmovdqu   (160+\off)(%rdi),%ymm9
vmovdqu   (192+\off)(%rdi),%ymm10
vmovdqu   (224+\off)(%rdi),%ymm11

vpaddd    %ymm4,%ymm8,%ymm4
vpaddd    %ymm5,%ymm9,%ymm5
vpaddd    %ymm6,%ymm10,%ymm6
vpaddd    %ymm7,%ymm11,%ymm7

vmovdqu   %ymm0,(\off)(%rdi)
vmovdqu   %ymm1,(32+\off)(%rdi)
vmovdqu   %ymm2,(64+\off)(%rdi)
vmovdqu   %ymm3,(96+\off)(%rdi)
vmovdqu   %ymm4,(128+\off)(%rdi)
vmovdqu   %ymm5,(160+\off)(%rdi)
vmovdqu   %ymm6,(192+\off)(%rdi)
vmovdqu   %ymm7,(224+\off)(%rdi)

.endm

.p2align 5   //instructs the assembler to align the following instruction or data on a boundary that is a power of 2 and here is equal to 2^5, or 32 bytes
.global cdecl(add_asm)
cdecl(add_asm):

accanswer       0
accanswer       256
accanswer       512
accanswer       768

ret




.macro accmaskanswer off
vmovdqu   (\off)(%rsi),%ymm0
vmovdqu   (32+\off)(%rsi),%ymm1
vmovdqu   (64+\off)(%rsi),%ymm2
vmovdqu   (96+\off)(%rsi),%ymm3
vmovdqu   (128+\off)(%rsi),%ymm4
vmovdqu   (160+\off)(%rsi),%ymm5
vmovdqu   (192+\off)(%rsi),%ymm6
vmovdqu   (224+\off)(%rsi),%ymm7
vmovdqu   (\off)(%rdi),%ymm8
vmovdqu   (32+\off)(%rdi),%ymm9
vmovdqu   (64+\off)(%rdi),%ymm10
vmovdqu   (96+\off)(%rdi),%ymm11

vpsubd    %ymm0,%ymm14,%ymm0
vpsubd    %ymm1,%ymm14,%ymm1
vpsubd    %ymm2,%ymm14,%ymm2
vpsubd    %ymm3,%ymm14,%ymm3

vpaddd    %ymm0,%ymm8,%ymm0
vpaddd    %ymm1,%ymm9,%ymm1
vpaddd    %ymm2,%ymm10,%ymm2
vpaddd    %ymm3,%ymm11,%ymm3

vmovdqu   (128+\off)(%rdi),%ymm8
vmovdqu   (160+\off)(%rdi),%ymm9
vmovdqu   (192+\off)(%rdi),%ymm10
vmovdqu   (224+\off)(%rdi),%ymm11

vpsubd    %ymm4,%ymm14,%ymm4
vpsubd    %ymm5,%ymm14,%ymm5
vpsubd    %ymm6,%ymm14,%ymm6
vpsubd    %ymm7,%ymm14,%ymm7

vpaddd    %ymm4,%ymm8,%ymm4
vpaddd    %ymm5,%ymm9,%ymm5
vpaddd    %ymm6,%ymm10,%ymm6
vpaddd    %ymm7,%ymm11,%ymm7

vmovdqu   %ymm0,(\off)(%rdi)
vmovdqu   %ymm1,(32+\off)(%rdi)
vmovdqu   %ymm2,(64+\off)(%rdi)
vmovdqu   %ymm3,(96+\off)(%rdi)
vmovdqu   %ymm4,(128+\off)(%rdi)
vmovdqu   %ymm5,(160+\off)(%rdi)
vmovdqu   %ymm6,(192+\off)(%rdi)
vmovdqu   %ymm7,(224+\off)(%rdi)

.endm

.p2align 5   //instructs the assembler to align the following instruction or data on a boundary that is a power of 2 and here is equal to 2^5, or 32 bytes
.global cdecl(addmask_asm)
cdecl(addmask_asm):

vpbroadcastd		_8xmasks(%rip),%ymm14

accmaskanswer       0
accmaskanswer       256
accmaskanswer       512
accmaskanswer       768

ret
