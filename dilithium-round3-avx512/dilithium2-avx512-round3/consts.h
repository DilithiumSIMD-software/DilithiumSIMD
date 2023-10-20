#ifndef CONSTS_H
#define CONSTS_H

#include "params.h"

#define _16XQ          0
#define _16XQINV       16
#define _16XDIV_QINV   32
#define _16XDIV        48
#define _ZETAS_QINV   64
#define _ZETAS        592
#define _ZETASINV_QINV 1120
#define _ZETASINV 1648
 #define _mQ 2176
/* The C ABI on MacOS exports all symbols with a leading
 * underscore. This means that any symbols we refer to from
 * C files (functions) can't be found, and all symbols we
 * refer to from ASM also can't be found.
 *
 * This define helps us get around this
 */

#define cdecl(s) s

#ifndef __ASSEMBLER__

#include "align.h"

typedef ALIGNED_INT32(2192) qdata_t;

extern const qdata_t qdata;

#endif
#endif
