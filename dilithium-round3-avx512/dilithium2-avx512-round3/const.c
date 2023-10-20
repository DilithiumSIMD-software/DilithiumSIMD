#include <stdint.h>
#include "params.h"

#define MASKSE 0X04040404


const uint32_t _16xeta[16]  __attribute__((aligned(32))) = {ETA, ETA, ETA, ETA, ETA, ETA, ETA, ETA,ETA, ETA, ETA, ETA, ETA, ETA, ETA, ETA};
const uint32_t _16xmasks[16]  __attribute__((aligned(32))) = {MASKSE, MASKSE, MASKSE, MASKSE, MASKSE, MASKSE, MASKSE, MASKSE,MASKSE, MASKSE, MASKSE, MASKSE, MASKSE, MASKSE, MASKSE, MASKSE};