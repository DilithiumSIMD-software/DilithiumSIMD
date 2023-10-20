#include <stdint.h>
#include "params.h"

#define MASKS1 0X04040404
#define MASKS2 0X040404


const uint32_t _8xeta[8]  __attribute__((aligned(32))) = {ETA, ETA, ETA, ETA, ETA, ETA, ETA, ETA};
const uint32_t _8xmask1[8]  __attribute__((aligned(32))) = {MASKS1, MASKS1, MASKS1, MASKS1, MASKS1, MASKS1, MASKS1, MASKS1};
const uint32_t _8xmask2[8]  __attribute__((aligned(32))) = {MASKS2, MASKS2, MASKS2, MASKS2, MASKS2, MASKS2, MASKS2, MASKS2};
