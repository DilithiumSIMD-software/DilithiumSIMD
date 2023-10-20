#ifndef SHUFFLE8_H
#define SHUFFLE8_H

void shufflelo8_avx(__m512i *c, const __m512i *a, const __m512i *b);
void shufflehi8_avx(__m512i *c, const __m512i *a, const __m512i *b);

#endif
