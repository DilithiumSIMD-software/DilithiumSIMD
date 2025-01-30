#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include "../randombytes.h"
#include "../sign.h"
#include <papi.h>
#include <stdio.h>
#include <time.h>
#define MLEN 59
#define NTESTS 10000
#define NUM_ITERATIONS 100000
// int main() {
//     unsigned int i, j;
//     int ret;
//     size_t mlen, smlen;
//     smlen = 3352;
//     uint8_t m[MLEN] = {0};
//     uint8_t sm[MLEN + CRYPTO_BYTES];
//     uint8_t m2[MLEN + CRYPTO_BYTES];
//     uint8_t pk[CRYPTO_PUBLICKEYBYTES];
//     uint8_t sk[CRYPTO_SECRETKEYBYTES];

//     // 生成随机消息
//     randombytes(m, MLEN);

//     // 生成密钥对
//     crypto_sign_keypair(pk, sk);

//     // 签名消息
//     crypto_sign(sm, &smlen, m, MLEN, sk);

//     // 验证签名
//     ret = crypto_sign_open(m2, &mlen, sm, smlen, pk);
//     return 0;
// }


void function_to_measure() {
    // 模拟要测量的代码
    unsigned int i, j;
    int ret;
    size_t mlen, smlen;
    smlen = 3352;
    uint8_t m[MLEN] = {0};
    uint8_t sm[MLEN + CRYPTO_BYTES];
    uint8_t m2[MLEN + CRYPTO_BYTES];
    uint8_t pk[CRYPTO_PUBLICKEYBYTES];
    uint8_t sk[CRYPTO_SECRETKEYBYTES];

    // 生成随机消息
    randombytes(m, MLEN);

    // 生成密钥对
    crypto_sign_keypair(pk, sk);

    // 签名消息
    crypto_sign(sm, &smlen, m, MLEN, sk);

    // 验证签名
    ret = crypto_sign_open(m2, &mlen, sm, smlen, pk);
}

int main() {
    struct timespec start, end;
    double total_time = 0.0;

    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        // 获取起始时间
        clock_gettime(CLOCK_MONOTONIC, &start);

        // 调用需要测量的函数
        function_to_measure();

        // 获取结束时间
        clock_gettime(CLOCK_MONOTONIC, &end);

        // 计算本次运行的时间差（以微秒为单位）
        long seconds = end.tv_sec - start.tv_sec;
        long nanoseconds = end.tv_nsec - start.tv_nsec;
        double elapsed = seconds * 1e6 + nanoseconds / 1e3; // 转换为微秒

        // 累加时间
        total_time += elapsed;
    }

    // 计算平均值
    double average_time = total_time / NUM_ITERATIONS;
    printf("Average Wall Time over %d iterations: %.2f μs\n", NUM_ITERATIONS, average_time);

    return 0;
}

// int main() {
//     // 初始化PAPI库
//     if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) {
//         fprintf(stderr, "PAPI initialization failed!\n");
//         return 1;
//     }

//     // 创建事件集
//     int event_set = PAPI_NULL;
//     if (PAPI_create_eventset(&event_set) != PAPI_OK) {
//         fprintf(stderr, "Error creating event set!\n");
//         return 1;
//     }

//     // 添加事件（总指令数，总周期数）
//     if (PAPI_add_event(event_set, PAPI_TOT_INS) != PAPI_OK) {
//         fprintf(stderr, "Error adding PAPI_TOT_INS event!\n");
//         return 1;
//     }

//     if (PAPI_add_event(event_set, PAPI_TOT_CYC) != PAPI_OK) {
//         fprintf(stderr, "Error adding PAPI_TOT_CYC event!\n");
//         return 1;
//     }

//     // 启动计数器
//     if (PAPI_start(event_set) != PAPI_OK) {
//         fprintf(stderr, "Error starting PAPI events!\n");
//         return 1;
//     }

//     // 执行程序 (加密计算)
//     unsigned int i, j;
//     int ret;
//     size_t mlen, smlen;
//     smlen = 3352;
//     uint8_t m[MLEN] = {0};
//     uint8_t sm[MLEN + CRYPTO_BYTES];
//     uint8_t m2[MLEN + CRYPTO_BYTES];
//     uint8_t pk[CRYPTO_PUBLICKEYBYTES];
//     uint8_t sk[CRYPTO_SECRETKEYBYTES];

//     // 生成随机消息
//     randombytes(m, MLEN);

//     // 生成密钥对
//     crypto_sign_keypair(pk, sk);

//     // 签名消息
//     crypto_sign(sm, &smlen, m, MLEN, sk);

//     // 验证签名
//     ret = crypto_sign_open(m2, &mlen, sm, smlen, pk);
//     if (ret != 0) {
//         fprintf(stderr, "Signature verification failed!\n");
//         return 1;
//     }

//     // 停止计数器
//     if (PAPI_stop(event_set, NULL) != PAPI_OK) {
//         fprintf(stderr, "Error stopping PAPI events!\n");
//         return 1;
//     }

//     // 读取计数器值
//     long long instructions, cycles;
//     if (PAPI_read(event_set, &instructions) != PAPI_OK) {
//         fprintf(stderr, "Error reading PAPI_TOT_INS!\n");
//         return 1;
//     }
//     if (PAPI_read(event_set, &cycles) != PAPI_OK) {
//         fprintf(stderr, "Error reading PAPI_TOT_CYC!\n");
//         return 1;
//     }

//     // 输出结果
//     printf("Instructions: %.2f\n", (double)instructions);
//     printf("Cycles: %.2f\n", (double)cycles);
//     if (cycles > 0) {
//         printf("IPC: %.2f\n", (double)instructions / cycles);
//     } else {
//         printf("IPC: N/A (cycles = 0)\n");
//     }

//     // 清理PAPI资源
//     PAPI_cleanup_eventset(event_set);
//     PAPI_destroy_eventset(&event_set);

//     return 0;
// }

// int test_scheme()
// {
//     unsigned int i, j;
//   int ret;
//   size_t mlen, smlen;
//   smlen = 3352;
//   uint8_t m[MLEN] = {0};
//   uint8_t sm[MLEN + CRYPTO_BYTES];
//   uint8_t m2[MLEN + CRYPTO_BYTES];
//   uint8_t pk[CRYPTO_PUBLICKEYBYTES];
//   uint8_t sk[CRYPTO_SECRETKEYBYTES];

//   for(i = 0; i < NTESTS; ++i) {
//     randombytes(m, MLEN);

//     crypto_sign_keypair(pk, sk);

//     crypto_sign(sm, &smlen, m, MLEN, sk);
//     ret = crypto_sign_open(m2, &mlen, sm, smlen, pk);

//     if(ret) {
//       fprintf(stderr, "Verification failed\n");
//       return -1;
//     }

//     if(mlen != MLEN) {
//       fprintf(stderr, "Message lengths don't match\n");
//       return -1;
//     }

//     for(j = 0; j < mlen; ++j) {
//       if(m[j] != m2[j]) {
//         fprintf(stderr, "Messages don't match\n");
//         return -1;
//       }
//     }

//     randombytes((uint8_t *)&j, sizeof(j));
//     do {
//       randombytes(m2, 1);
//     } while(!m2[0]);
//     sm[j % CRYPTO_BYTES] += m2[0];
//     ret = crypto_sign_open(m2, &mlen, sm, smlen, pk);
//     if(!ret) {
//       fprintf(stderr, "Trivial forgeries possible\n");
//       return -1;
//     }
//   }

//   printf("CRYPTO_PUBLICKEYBYTES = %d\n", CRYPTO_PUBLICKEYBYTES);
//   printf("CRYPTO_SECRETKEYBYTES = %d\n", CRYPTO_SECRETKEYBYTES);
//   printf("CRYPTO_BYTES = %d\n", CRYPTO_BYTES);
//   return 0;
// }

