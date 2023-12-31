#include "SHA-3_256par.h"				


#define ROT(x,y)  _mm512_rol_epi64(x,y)
//#define ROT(x,y)  _mm256_or_si256(_mm256_slli_epi64(x,y),_mm256_srli_epi64(x,(64-y)))
#define ROT1(x)  _mm512_rol_epi64(x,0x1)
//#define ROT1(x)  _mm256_or_si256(_mm256_add_epi64(x,x),_mm256_srli_epi64(x,(63)))

#define load(x,m,d) 	  	x = _mm512_loadu_si512((__m512i*)m+d)
#define store(x,d,y)		_mm256_storeu_si256((__m256i*)x+d,y);
//#define blend_32(x,y,z,d) 	x = _mm256_blend_epi32(y,z,d)
//#define permut_64(x,y,d)  	x = _mm256_permute4x64_epi64(y,d)
//#define xor_pd(x,y,z) 	  	x = _mm256_xor_pd(y,z)
#define xor(x,y,z) 	  	x = _mm512_xor_si512(y,z)
//#define andnot(y,z)		_mm256_andnot_si256(y,z)
#define unpackLO_64(x,y,z)	x = _mm512_unpacklo_epi64(y,z)
#define unpackHI_64(x,y,z)	x = _mm512_unpackhi_epi64(y,z)
//#define extract_128(x,y,d)	x = _mm256_extracti128_si256(y,d)
#define insert_128(x,y,z,d)	x = _mm256_inserti128_si256(y,z,d)
#define set_zero_256(x)		x = _mm512_setzero_si512()

#define load_128(x,m,d) 	x = _mm_loadu_si128((__m128i*)m+d)
#define store_128(x,d,y)	_mm_storeu_si128((__m128i*)x+d,y);
#define unpackLO_128_64(x,y,z)	x = _mm_unpacklo_epi64(y,z)
#define unpackHI_128_64(x,y,z)	x = _mm_unpackhi_epi64(y,z)
#define cast_to_256(x,y)	x = _mm256_castsi128_si256(y)

#define init_S_zeros(y)		for(y=0;y<25;y++){\
				  set_zero_256(s[y]);}

#define loadM(y,p,temp)		\
				for(y=0;y<p;y++){\
					for(i=0;i<8;i++)\
                                                load(t[i],temp[i],y);\
                                        \
                                        for(i=0;i<8;i=i+2){\
                                                unpackLO_64(r[i],  t[i],t[i+1]);\
                                                unpackHI_64(r[i+1],t[i],t[i+1]);\
                                        }\
                                        for(i=0;i<8;i=i+4){\
                                                t[i]   = _mm512_permutex2var_epi64(r[i],l0,r[i+2]);/*d2 c2 b2 a2 d0 c0 b0 a0*/\
                                                t[i+1] = _mm512_permutex2var_epi64(r[i],l1,r[i+2]);/*d6 c6 b6 a6 d4 c4 b4 a4*/\
                                                t[i+2] = _mm512_permutex2var_epi64(r[i+1],l0,r[i+3]);/*d5 c5 b5 a5 d1 c1 b1 a1*/\
                                                t[i+3] = _mm512_permutex2var_epi64(r[i+1],l1,r[i+3]);/*d7 c7 b7 a7 d3 c3 b3 a3*/\
                                        }\
                                        for(i=0;i<2;i++){\
                                                r[(i<<2)]   = _mm512_permutex2var_epi64(t[i],l2,t[i+4]);/*h2 g2 f2 e2 h0 g0 f0 e0*/\
                                                r[(i<<2)+1] = _mm512_permutex2var_epi64(t[i+2],l2,t[i+6]);/*h6 g6 f6 e6 h4 g4 f4 e4*/\
                                                r[(i<<2)+2] = _mm512_permutex2var_epi64(t[i],l3,t[i+4]);/*h5 g5 f5 e5 h1 g1 f1 e1*/\
                                                r[(i<<2)+3] = _mm512_permutex2var_epi64(t[i+2],l3,t[i+6]);/*h7 g7 f7 e7 h3 g3 g3 e3*/\
                                        }\
                                        for(i=0;i<8;i++)\
                                                xor(s[i+(y*8)],s[i+(y*8)],r[i]);\
                                        \
				}\
				if(rsiz == 104 || rsiz == 168){\
					for(i=0;i<8;i++)\
                                                load(t[i],temp[i],y);\
                                        \
                                        for(i=0;i<8;i=i+2){\
                                                unpackLO_64(r[i],  t[i],t[i+1]);\
                                                unpackHI_64(r[i+1],t[i],t[i+1]);\
                                        }\
                                        for(i=0;i<8;i=i+4){\
                                                t[i]   = _mm512_permutex2var_epi64(r[i],l0,r[i+2]);/*d2 c2 b2 a2 d0 c0 b0 a0*/\
                                                t[i+1] = _mm512_permutex2var_epi64(r[i],l1,r[i+2]);/*d6 c6 b6 a6 d4 c4 b4 a4*/\
                                                t[i+2] = _mm512_permutex2var_epi64(r[i+1],l0,r[i+3]);/*d5 c5 b5 a5 d1 c1 b1 a1*/\
                                                t[i+3] = _mm512_permutex2var_epi64(r[i+1],l1,r[i+3]);/*d7 c7 b7 a7 d3 c3 b3 a3*/\
                                        }\
                                        for(i=0;i<2;i++){\
                                                r[(i<<2)]   = _mm512_permutex2var_epi64(t[i],l2,t[i+4]);/*h2 g2 f2 e2 h0 g0 f0 e0*/\
                                                r[(i<<2)+1] = _mm512_permutex2var_epi64(t[i+2],l2,t[i+6]);/*h6 g6 f6 e6 h4 g4 f4 e4*/\
                                                r[(i<<2)+2] = _mm512_permutex2var_epi64(t[i],l3,t[i+4]);/*h5 g5 f5 e5 h1 g1 f1 e1*/\
                                                r[(i<<2)+3] = _mm512_permutex2var_epi64(t[i+2],l3,t[i+6]);/*h7 g7 f7 e7 h3 g3 g3 e3*/\
                                        }\
                                        for(i=0;i<5;i++)\
                                                xor(s[i+(y*8)],s[i+(y*8)],r[i]);\
                                        \
				}else{\
					load_128(auxl[0],temp[0],y*4);\
					load_128(auxl[1],temp[1],y*4);\
					load_128(auxl[2],temp[2],y*4);\
					load_128(auxl[3],temp[3],y*4);\
					\
					unpackLO_128_64(auxl[4],auxl[0],auxl[1]);\
					unpackLO_128_64(auxl[5],auxl[2],auxl[3]);\
					cast_to_256(auxM,auxl[4]);\
					insert_128(auxM,auxM,auxl[5],1);\
					aux[0] = _mm512_castsi256_si512(auxM);\
					\
					if(rsiz == 144){\
					unpackHI_128_64(auxl[4],auxl[0],auxl[1]);\
						unpackHI_128_64(auxl[5],auxl[2],auxl[3]);\
						cast_to_256(auxM,auxl[4]);\
						insert_128(auxM,auxM,auxl[5],1);\
						r[0] = _mm512_castsi256_si512(auxM);\
					}\
					\
					load_128(auxl[0],temp[4],y*4);\
					load_128(auxl[1],temp[5],y*4);\
					load_128(auxl[2],temp[6],y*4);\
					load_128(auxl[3],temp[7],y*4);\
					\
					if(rsiz == 144){\
						unpackHI_128_64(auxl[4],auxl[0],auxl[1]);\
						unpackHI_128_64(auxl[5],auxl[2],auxl[3]);\
						cast_to_256(auxM,auxl[4]);\
						insert_128(auxM,auxM,auxl[5],1);\
						r[0] = _mm512_inserti64x4(r[0],auxM,1);\
						xor(s[1+(y*8)],s[1+(y*8)],r[0]);\
					}\
					unpackLO_128_64(auxl[4],auxl[0],auxl[1]);\
					unpackLO_128_64(auxl[5],auxl[2],auxl[3]);\
					cast_to_256(auxM,auxl[4]);\
					insert_128(auxM,auxM,auxl[5],1);\
					aux[0] = _mm512_inserti64x4(aux[0],auxM,1);\
					xor(s[0+(y*8)],s[0+(y*8)],aux[0]);\
				}
				


#define back(y,p,temp)		\
			for(y=0;y<6;y++){\
				unpackLO_64(aux[0],s[0+(y*4)],s[1+(y*4)]);   /*g1 g0 e1 e0 c1 c0 a1 a0*/\
				unpackHI_64(aux[1],s[0+(y*4)],s[1+(y*4)]);   /*h1 h0 f1 f0 d1 d0 b1 b0*/\
				unpackLO_64(aux[2],s[2+(y*4)],s[3+(y*4)]);   /*g3 g2 e3 e2 c3 c2 a3 a2*/\
				unpackHI_64(aux[3],s[2+(y*4)],s[3+(y*4)]);   /*h3 h2 f3 f2 d3 d2 b3 b2*/\
				\
				aux[4] = _mm512_permutex2var_epi64(aux[0],l0,aux[2]);/*c3 c2 c1 c0 a3 a2 a1 a0*/\
				aux[5] = _mm512_permutex2var_epi64(aux[0],l1,aux[2]);/*g3 g2 g1 g0 e3 e2 e1 e0*/\
				aux[6] = _mm512_permutex2var_epi64(aux[1],l0,aux[3]);/*d3 d2 d1 d0 b3 b2 b1 b0*/\
				aux[7] = _mm512_permutex2var_epi64(aux[1],l1,aux[3]);/*h3 h2 h1 h0 f3 f2 f1 f0*/\
				\
				store(temp[0],y,_mm512_castsi512_si256(aux[4]));\
				store(temp[1],y,_mm512_castsi512_si256(aux[6]));\
  				store(temp[2],y,_mm512_extracti64x4_epi64(aux[4],1));\
	  			store(temp[3],y,_mm512_extracti64x4_epi64(aux[6],1));\
				\
				store(temp[4],y,_mm512_castsi512_si256(aux[5]));\
				store(temp[5],y,_mm512_castsi512_si256(aux[7]));\
  				store(temp[6],y,_mm512_extracti64x4_epi64(aux[5],1));\
	  			store(temp[7],y,_mm512_extracti64x4_epi64(aux[7],1));\
			}


#define keccakf(s)	\
			for(i=0;i<24;i=i+2){\
      \
	aux[0] = _mm512_ternarylogic_epi64(s[0],s[5],s[10],0x96);\
	aux[0] = _mm512_ternarylogic_epi64(aux[0],s[15],s[20],0x96);\
	aux[2] = _mm512_ternarylogic_epi64(s[2],s[7],s[12],0x96);\
	aux[2] = _mm512_ternarylogic_epi64(aux[2],s[17],s[22],0x96);\
	aux[1] = _mm512_ternarylogic_epi64(s[1],s[6],s[11],0x96);\
	aux[1] = _mm512_ternarylogic_epi64(aux[1],s[16],s[21],0x96);\
	aux[3] = _mm512_ternarylogic_epi64(s[3],s[8],s[13],0x96);\
	aux[3] = _mm512_ternarylogic_epi64(aux[3],s[18],s[23],0x96);\
	aux[4] = _mm512_ternarylogic_epi64(s[4],s[9],s[14],0x96);\
	aux[4] = _mm512_ternarylogic_epi64(aux[4],s[19],s[24],0x96);\
	\
	aux[6] = ROT1(aux[2]);/*aux[6] = d1*/\
	aux[7] = ROT1(aux[3]);/*aux[7] = d2*/\
	aux[9] = ROT1(aux[0]);/*aux[9] = d4*/\
	aux[5] = ROT1(aux[1]);/*aux[5] = d0*/\
	aux[8] = ROT1(aux[4]);/*aux[8] = d3*/\
	\
	s[0] = _mm512_ternarylogic_epi64(s[0],aux[4],aux[5],0x96);\
	s[5] = _mm512_ternarylogic_epi64(s[5],aux[4],aux[5],0x96);\
	s[10] = _mm512_ternarylogic_epi64(s[10],aux[4],aux[5],0x96);\
	s[15] = _mm512_ternarylogic_epi64(s[15],aux[4],aux[5],0x96);\
	s[20] = _mm512_ternarylogic_epi64(s[20],aux[4],aux[5],0x96);\
	\
	s[1] = _mm512_ternarylogic_epi64(s[1],aux[6],aux[0],0x96);\
	s[6] = _mm512_ternarylogic_epi64(s[6],aux[6],aux[0],0x96);\
	s[11] = _mm512_ternarylogic_epi64(s[11],aux[6],aux[0],0x96);\
	s[16] = _mm512_ternarylogic_epi64(s[16],aux[6],aux[0],0x96);\
	s[21] = _mm512_ternarylogic_epi64(s[21],aux[6],aux[0],0x96);\
	\
	s[2] = _mm512_ternarylogic_epi64(s[2],aux[7],aux[1],0x96);\
	s[7] = _mm512_ternarylogic_epi64(s[7],aux[7],aux[1],0x96);\
	s[12] = _mm512_ternarylogic_epi64(s[12],aux[7],aux[1],0x96);\
	s[17] = _mm512_ternarylogic_epi64(s[17],aux[7],aux[1],0x96);\
	s[22] = _mm512_ternarylogic_epi64(s[22],aux[7],aux[1],0x96);\
	\
	s[3] = _mm512_ternarylogic_epi64(s[3],aux[8],aux[2],0x96);\
	s[8] = _mm512_ternarylogic_epi64(s[8],aux[8],aux[2],0x96);\
	s[13] = _mm512_ternarylogic_epi64(s[13],aux[8],aux[2],0x96);\
	s[18] = _mm512_ternarylogic_epi64(s[18],aux[8],aux[2],0x96);\
	s[23] = _mm512_ternarylogic_epi64(s[23],aux[8],aux[2],0x96);\
	\
	s[4] = _mm512_ternarylogic_epi64(s[4],aux[9],aux[3],0x96);\
	s[9] = _mm512_ternarylogic_epi64(s[9],aux[9],aux[3],0x96);\
	s[14] = _mm512_ternarylogic_epi64(s[14],aux[9],aux[3],0x96);\
	s[19] = _mm512_ternarylogic_epi64(s[19],aux[9],aux[3],0x96);\
	s[24] = _mm512_ternarylogic_epi64(s[24],aux[9],aux[3],0x96);\
	\
	aux[1] = ROT(s[6],0x2C);\
	aux[2] = ROT(s[12],0x2B);\
	aux[3] = ROT(s[18],0x15);\
	aux[4] = ROT(s[24],0x0E);\
	\
	\
	s[18] = _mm512_ternarylogic_epi64(aux[4],s[0],aux[1],0xD2);/*s[18] = s4*/\
	s[6] = _mm512_ternarylogic_epi64(aux[3],aux[4],s[0],0xD2);/*s[6] = s3*/\
	s[0] = _mm512_ternarylogic_epi64(s[0],aux[1],aux[2],0xD2);/*s[0] = s0*/\
	s[12] = _mm512_ternarylogic_epi64(aux[1],aux[2],aux[3],0xD2);/*s[12] = s1*/\
	load(aux[6],RC,i);\
	s[24] = _mm512_ternarylogic_epi64(aux[2],aux[3],aux[4],0xD2);/*s[24] = s2*/\
	\
	xor(s[0],s[0],aux[6]);\
	\
	\
	aux[0] = ROT(s[3],0x1C);\
	aux[1] = ROT(s[9],0x14);\
	aux[2] = ROT(s[10],0x3);\
	aux[3] = ROT(s[16],0x2D);\
	aux[4] = ROT(s[22],0x3D);\
	\
	s[16] = _mm512_ternarylogic_epi64(aux[0],aux[1],aux[2],0xD2);/*s[16] = s5*/\
	s[3]  = _mm512_ternarylogic_epi64(aux[1],aux[2],aux[3],0xD2);/*s[3] = s6*/\
	s[10] = _mm512_ternarylogic_epi64(aux[2],aux[3],aux[4],0xD2);/*s[10] = s7*/\
	s[22] = _mm512_ternarylogic_epi64(aux[3],aux[4],aux[0],0xD2);/*s[22] = s8*/\
	s[9]  = _mm512_ternarylogic_epi64(aux[4],aux[0],aux[1],0xD2);/*s[9] = s9*/\
	\
	aux[0] = ROT(s[1],0x01);\
	aux[1] = ROT(s[7],0x06);\
	aux[2] = ROT(s[13],0x19);\
	aux[3] = ROT(s[19],0x8);\
	aux[4] = ROT(s[20],0x12);\
	\
	s[7]   = _mm512_ternarylogic_epi64(aux[0],aux[1],aux[2],0xD2);/*s[7] = s10*/\
	s[19]  = _mm512_ternarylogic_epi64(aux[1],aux[2],aux[3],0xD2);/*s[19] = s11*/\
	s[1]   = _mm512_ternarylogic_epi64(aux[2],aux[3],aux[4],0xD2);/*s[1] = s12*/\
	s[13]  = _mm512_ternarylogic_epi64(aux[3],aux[4],aux[0],0xD2);/*s[13] = s13*/\
	s[20]  = _mm512_ternarylogic_epi64(aux[4],aux[0],aux[1],0xD2);/*s[20] = s14*/\
	\
	aux[0] = ROT(s[4],0x1B);\
	aux[1] = ROT(s[5],0x24);\
	aux[2] = ROT(s[11],0xA);\
	aux[3] = ROT(s[17],0xF);\
	aux[4] = ROT(s[23],0x38);\
	\
	s[23] = _mm512_ternarylogic_epi64(aux[0],aux[1],aux[2],0xD2);/*s[23] = s15*/\
	s[5]  = _mm512_ternarylogic_epi64(aux[1],aux[2],aux[3],0xD2);/*s[5] = s16*/\
	s[17] = _mm512_ternarylogic_epi64(aux[2],aux[3],aux[4],0xD2);/*s[17] = s17*/\
	s[4]  = _mm512_ternarylogic_epi64(aux[3],aux[4],aux[0],0xD2);/*s[4] = s18*/\
	s[11] = _mm512_ternarylogic_epi64(aux[4],aux[0],aux[1],0xD2);/*s[11] = s19*/\
	\
	aux[0] = ROT(s[2],0x3E);\
	aux[1] = ROT(s[8],0x37);\
	aux[2] = ROT(s[14],0x27);\
	aux[3] = ROT(s[15],0x29);\
	aux[4] = ROT(s[21],0x2);\
	\
	s[14] = _mm512_ternarylogic_epi64(aux[0],aux[1],aux[2],0xD2);/*s[14] = s20*/\
	s[21] = _mm512_ternarylogic_epi64(aux[1],aux[2],aux[3],0xD2);/*s[21] = s21*/\
	s[8]  = _mm512_ternarylogic_epi64(aux[2],aux[3],aux[4],0xD2);/*s[8] = s22*/\
	s[15] = _mm512_ternarylogic_epi64(aux[3],aux[4],aux[0],0xD2);/*s[15] = s23*/\
	s[2]  = _mm512_ternarylogic_epi64(aux[4],aux[0],aux[1],0xD2);/*s[2] = s24*/\
	\
	/*new round*/\
	\
	aux[0] = _mm512_ternarylogic_epi64(s[0],s[16],s[7],0x96);\
	aux[0] = _mm512_ternarylogic_epi64(aux[0],s[23],s[14],0x96);\
	aux[2] = _mm512_ternarylogic_epi64(s[24],s[10],s[1],0x96);\
	aux[2] = _mm512_ternarylogic_epi64(aux[2],s[17],s[8],0x96);\
	aux[1] = _mm512_ternarylogic_epi64(s[12],s[3],s[19],0x96);\
	aux[1] = _mm512_ternarylogic_epi64(aux[1],s[5],s[21],0x96);\
	aux[3] = _mm512_ternarylogic_epi64(s[6],s[22],s[13],0x96);\
	aux[3] = _mm512_ternarylogic_epi64(aux[3],s[4],s[15],0x96);\
	aux[4] = _mm512_ternarylogic_epi64(s[18],s[9],s[20],0x96);\
	aux[4] = _mm512_ternarylogic_epi64(aux[4],s[11],s[2],0x96);\
	\
	aux[6] = ROT(aux[2],0x01);/*aux[6] = d1*/\
	aux[7] = ROT(aux[3],0x01);/*aux[7] = d2*/\
	aux[9] = ROT(aux[0],0x01);/*aux[9] = d4*/\
	aux[5] = ROT(aux[1],0x01);/*aux[5] = d0*/\
	aux[8] = ROT(aux[4],0x01);/*aux[8] = d3*/\
	\
	\
	s[0] = _mm512_ternarylogic_epi64(s[0],aux[5],aux[4],0x96);\
	s[16] = _mm512_ternarylogic_epi64(s[16],aux[5],aux[4],0x96);\
	s[7] = _mm512_ternarylogic_epi64(s[7],aux[5],aux[4],0x96);\
	s[23] = _mm512_ternarylogic_epi64(s[23],aux[5],aux[4],0x96);\
	s[14] = _mm512_ternarylogic_epi64(s[14],aux[5],aux[4],0x96);\
	\
	s[12] = _mm512_ternarylogic_epi64(s[12],aux[6],aux[0],0x96);\
	s[3] = _mm512_ternarylogic_epi64(s[3],aux[6],aux[0],0x96);\
	s[19] = _mm512_ternarylogic_epi64(s[19],aux[6],aux[0],0x96);\
	s[5] = _mm512_ternarylogic_epi64(s[5],aux[6],aux[0],0x96);\
	s[21] = _mm512_ternarylogic_epi64(s[21],aux[6],aux[0],0x96);\
	\
	s[24] = _mm512_ternarylogic_epi64(s[24],aux[7],aux[1],0x96);\
	s[10] = _mm512_ternarylogic_epi64(s[10],aux[7],aux[1],0x96);\
	s[1] = _mm512_ternarylogic_epi64(s[1],aux[7],aux[1],0x96);\
	s[17] = _mm512_ternarylogic_epi64(s[17],aux[7],aux[1],0x96);\
	s[8] = _mm512_ternarylogic_epi64(s[8],aux[7],aux[1],0x96);\
	\
	s[6] = _mm512_ternarylogic_epi64(s[6],aux[8],aux[2],0x96);\
	s[22] = _mm512_ternarylogic_epi64(s[22],aux[8],aux[2],0x96);\
	s[13] = _mm512_ternarylogic_epi64(s[13],aux[8],aux[2],0x96);\
	s[4] = _mm512_ternarylogic_epi64(s[4],aux[8],aux[2],0x96);\
	s[15] = _mm512_ternarylogic_epi64(s[15],aux[8],aux[2],0x96);\
	\
	s[18] = _mm512_ternarylogic_epi64(s[18],aux[9],aux[3],0x96);\
	s[9] = _mm512_ternarylogic_epi64(s[9],aux[9],aux[3],0x96);\
	s[20] = _mm512_ternarylogic_epi64(s[20],aux[9],aux[3],0x96);\
	s[11] = _mm512_ternarylogic_epi64(s[11],aux[9],aux[3],0x96);\
	s[2] = _mm512_ternarylogic_epi64(s[2],aux[9],aux[3],0x96);\
	\
	\
	aux[1] = ROT(s[3],0x2C);\
	aux[2] = ROT(s[1],0x2B);\
	aux[3] = ROT(s[4],0x15);\
	aux[4] = ROT(s[2],0x0E);\
	\
	s[3] = _mm512_ternarylogic_epi64(aux[3],aux[4],s[0],0xD2);/*s[3] = s3*/\
	s[4] = _mm512_ternarylogic_epi64(aux[4],s[0],aux[1],0xD2);/*s[4] = s4*/\
	s[0] = _mm512_ternarylogic_epi64(s[0],aux[1],aux[2],0xD2);/*s[0] = s0*/\
	s[1] = _mm512_ternarylogic_epi64(aux[1],aux[2],aux[3],0xD2);/*s[1] = s1*/\
	load(aux[6],RC,(i+1));\
	s[2] = _mm512_ternarylogic_epi64(aux[2],aux[3],aux[4],0xD2);/*s[2] = s2*/\
	\
	xor(s[0],s[0],aux[6]);\
	\
	aux[0] = ROT(s[6],0x1C);\
	aux[1] = ROT(s[9],0x14);\
	aux[2] = ROT(s[7],0x3);\
	aux[3] = ROT(s[5],0x2D);\
	aux[4] = ROT(s[8],0x3D);\
	\
	s[5] = _mm512_ternarylogic_epi64(aux[0],aux[1],aux[2],0xD2);/*s[5] = s5*/\
	s[6] = _mm512_ternarylogic_epi64(aux[1],aux[2],aux[3],0xD2);/*s[6] = s6*/\
	s[7] = _mm512_ternarylogic_epi64(aux[2],aux[3],aux[4],0xD2);/*s[7] = s7*/\
	s[8] = _mm512_ternarylogic_epi64(aux[3],aux[4],aux[0],0xD2);/*s[8] = s8*/\
	s[9] = _mm512_ternarylogic_epi64(aux[4],aux[0],aux[1],0xD2);/*s[9] = s9*/\
	\
	aux[0] = ROT(s[12],0x1);\
	aux[1] = ROT(s[10],0x6);\
	aux[2] = ROT(s[13],0x19);\
	aux[3] = ROT(s[11],0x8);\
	aux[4] = ROT(s[14],0x12);\
	\
	s[10]  = _mm512_ternarylogic_epi64(aux[0],aux[1],aux[2],0xD2);/*s[10] = s10*/\
	s[11]  = _mm512_ternarylogic_epi64(aux[1],aux[2],aux[3],0xD2);/*s[11] = s11*/\
	s[12]  = _mm512_ternarylogic_epi64(aux[2],aux[3],aux[4],0xD2);/*s[12] = s12*/\
	s[13]  = _mm512_ternarylogic_epi64(aux[3],aux[4],aux[0],0xD2);/*s[13] = s13*/\
	s[14]  = _mm512_ternarylogic_epi64(aux[4],aux[0],aux[1],0xD2);/*s[14] = s14*/\
	\
	aux[0] = ROT(s[18],0x1B);\
	aux[1] = ROT(s[16],0x24);\
	aux[2] = ROT(s[19],0xA);\
	aux[3] = ROT(s[17],0xF);\
	aux[4] = ROT(s[15],0x38);\
	\
	s[15] = _mm512_ternarylogic_epi64(aux[0],aux[1],aux[2],0xD2);/*s[15] = s15*/\
	s[16]  = _mm512_ternarylogic_epi64(aux[1],aux[2],aux[3],0xD2);/*s[16] = s16*/\
	s[17] = _mm512_ternarylogic_epi64(aux[2],aux[3],aux[4],0xD2);/*s[17] = s17*/\
	s[18]  = _mm512_ternarylogic_epi64(aux[3],aux[4],aux[0],0xD2);/*s[18] = s18*/\
	s[19] = _mm512_ternarylogic_epi64(aux[4],aux[0],aux[1],0xD2);/*s[19] = s19*/\
	\
	aux[0] = ROT(s[24],0x3E);\
	aux[1] = ROT(s[22],0x37);\
	aux[2] = ROT(s[20],0x27);\
	aux[3] = ROT(s[23],0x29);\
	aux[4] = ROT(s[21],0x2);\
	\
	s[20] = _mm512_ternarylogic_epi64(aux[0],aux[1],aux[2],0xD2);/*s[20] = s20*/\
	s[21] = _mm512_ternarylogic_epi64(aux[1],aux[2],aux[3],0xD2);/*s[21] = s21*/\
	s[22] = _mm512_ternarylogic_epi64(aux[2],aux[3],aux[4],0xD2);/*s[22] = s22*/\
	s[23] = _mm512_ternarylogic_epi64(aux[3],aux[4],aux[0],0xD2);/*s[23] = s23*/\
	s[24] = _mm512_ternarylogic_epi64(aux[4],aux[0],aux[1],0xD2);/*s[24] = s24*/\
	}\
	\

void keccakF8x(__m512i state[25])
{
     __m512i aux[12];
   int i;
const ALIGN uint64_t RC[200] = {
    0x0000000000000001,0x0000000000000001,0x0000000000000001,0x0000000000000001, 0x0000000000000001,0x0000000000000001,0x0000000000000001,0x0000000000000001,
    0x0000000000008082,0x0000000000008082,0x0000000000008082,0x0000000000008082, 0x0000000000008082,0x0000000000008082,0x0000000000008082,0x0000000000008082,
    0x800000000000808A,0x800000000000808A,0x800000000000808A,0x800000000000808A, 0x800000000000808A,0x800000000000808A,0x800000000000808A,0x800000000000808A,
    0x8000000080008000,0x8000000080008000,0x8000000080008000,0x8000000080008000, 0x8000000080008000,0x8000000080008000,0x8000000080008000,0x8000000080008000,
    0x000000000000808B,0x000000000000808B,0x000000000000808B,0x000000000000808B, 0x000000000000808B,0x000000000000808B,0x000000000000808B,0x000000000000808B,
    0x0000000080000001,0x0000000080000001,0x0000000080000001,0x0000000080000001, 0x0000000080000001,0x0000000080000001,0x0000000080000001,0x0000000080000001,
    0x8000000080008081,0x8000000080008081,0x8000000080008081,0x8000000080008081, 0x8000000080008081,0x8000000080008081,0x8000000080008081,0x8000000080008081,
    0x8000000000008009,0x8000000000008009,0x8000000000008009,0x8000000000008009, 0x8000000000008009,0x8000000000008009,0x8000000000008009,0x8000000000008009,
    0x000000000000008A,0x000000000000008A,0x000000000000008A,0x000000000000008A, 0x000000000000008A,0x000000000000008A,0x000000000000008A,0x000000000000008A,
    0x0000000000000088,0x0000000000000088,0x0000000000000088,0x0000000000000088, 0x0000000000000088,0x0000000000000088,0x0000000000000088,0x0000000000000088,
    0x0000000080008009,0x0000000080008009,0x0000000080008009,0x0000000080008009, 0x0000000080008009,0x0000000080008009,0x0000000080008009,0x0000000080008009,
    0x000000008000000A,0x000000008000000A,0x000000008000000A,0x000000008000000A, 0x000000008000000A,0x000000008000000A,0x000000008000000A,0x000000008000000A,
    0x000000008000808B,0x000000008000808B,0x000000008000808B,0x000000008000808B, 0x000000008000808B,0x000000008000808B,0x000000008000808B,0x000000008000808B,
    0x800000000000008B,0x800000000000008B,0x800000000000008B,0x800000000000008B, 0x800000000000008B,0x800000000000008B,0x800000000000008B,0x800000000000008B,
    0x8000000000008089,0x8000000000008089,0x8000000000008089,0x8000000000008089, 0x8000000000008089,0x8000000000008089,0x8000000000008089,0x8000000000008089,
    0x8000000000008003,0x8000000000008003,0x8000000000008003,0x8000000000008003, 0x8000000000008003,0x8000000000008003,0x8000000000008003,0x8000000000008003,
    0x8000000000008002,0x8000000000008002,0x8000000000008002,0x8000000000008002, 0x8000000000008002,0x8000000000008002,0x8000000000008002,0x8000000000008002,
    0x8000000000000080,0x8000000000000080,0x8000000000000080,0x8000000000000080, 0x8000000000000080,0x8000000000000080,0x8000000000000080,0x8000000000000080,
    0x000000000000800A,0x000000000000800A,0x000000000000800A,0x000000000000800A, 0x000000000000800A,0x000000000000800A,0x000000000000800A,0x000000000000800A,
    0x800000008000000A,0x800000008000000A,0x800000008000000A,0x800000008000000A, 0x800000008000000A,0x800000008000000A,0x800000008000000A,0x800000008000000A,
    0x8000000080008081,0x8000000080008081,0x8000000080008081,0x8000000080008081, 0x8000000080008081,0x8000000080008081,0x8000000080008081,0x8000000080008081,
    0x8000000000008080,0x8000000000008080,0x8000000000008080,0x8000000000008080, 0x8000000000008080,0x8000000000008080,0x8000000000008080,0x8000000000008080,
    0x0000000080000001,0x0000000080000001,0x0000000080000001,0x0000000080000001, 0x0000000080000001,0x0000000080000001,0x0000000080000001,0x0000000080000001,
    0x8000000080008008,0x8000000080008008,0x8000000080008008,0x8000000080008008, 0x8000000080008008,0x8000000080008008,0x8000000080008008,0x8000000080008008
  };

    keccakf(state); 
}

int keccak(char **in_e, int inlen, uint8_t **md, int ra, int out)
{
   __m512i aux[12];
   int i;
const ALIGN uint64_t RC[200] = {
    0x0000000000000001,0x0000000000000001,0x0000000000000001,0x0000000000000001, 0x0000000000000001,0x0000000000000001,0x0000000000000001,0x0000000000000001,
    0x0000000000008082,0x0000000000008082,0x0000000000008082,0x0000000000008082, 0x0000000000008082,0x0000000000008082,0x0000000000008082,0x0000000000008082,
    0x800000000000808A,0x800000000000808A,0x800000000000808A,0x800000000000808A, 0x800000000000808A,0x800000000000808A,0x800000000000808A,0x800000000000808A,
    0x8000000080008000,0x8000000080008000,0x8000000080008000,0x8000000080008000, 0x8000000080008000,0x8000000080008000,0x8000000080008000,0x8000000080008000,
    0x000000000000808B,0x000000000000808B,0x000000000000808B,0x000000000000808B, 0x000000000000808B,0x000000000000808B,0x000000000000808B,0x000000000000808B,
    0x0000000080000001,0x0000000080000001,0x0000000080000001,0x0000000080000001, 0x0000000080000001,0x0000000080000001,0x0000000080000001,0x0000000080000001,
    0x8000000080008081,0x8000000080008081,0x8000000080008081,0x8000000080008081, 0x8000000080008081,0x8000000080008081,0x8000000080008081,0x8000000080008081,
    0x8000000000008009,0x8000000000008009,0x8000000000008009,0x8000000000008009, 0x8000000000008009,0x8000000000008009,0x8000000000008009,0x8000000000008009,
    0x000000000000008A,0x000000000000008A,0x000000000000008A,0x000000000000008A, 0x000000000000008A,0x000000000000008A,0x000000000000008A,0x000000000000008A,
    0x0000000000000088,0x0000000000000088,0x0000000000000088,0x0000000000000088, 0x0000000000000088,0x0000000000000088,0x0000000000000088,0x0000000000000088,
    0x0000000080008009,0x0000000080008009,0x0000000080008009,0x0000000080008009, 0x0000000080008009,0x0000000080008009,0x0000000080008009,0x0000000080008009,
    0x000000008000000A,0x000000008000000A,0x000000008000000A,0x000000008000000A, 0x000000008000000A,0x000000008000000A,0x000000008000000A,0x000000008000000A,
    0x000000008000808B,0x000000008000808B,0x000000008000808B,0x000000008000808B, 0x000000008000808B,0x000000008000808B,0x000000008000808B,0x000000008000808B,
    0x800000000000008B,0x800000000000008B,0x800000000000008B,0x800000000000008B, 0x800000000000008B,0x800000000000008B,0x800000000000008B,0x800000000000008B,
    0x8000000000008089,0x8000000000008089,0x8000000000008089,0x8000000000008089, 0x8000000000008089,0x8000000000008089,0x8000000000008089,0x8000000000008089,
    0x8000000000008003,0x8000000000008003,0x8000000000008003,0x8000000000008003, 0x8000000000008003,0x8000000000008003,0x8000000000008003,0x8000000000008003,
    0x8000000000008002,0x8000000000008002,0x8000000000008002,0x8000000000008002, 0x8000000000008002,0x8000000000008002,0x8000000000008002,0x8000000000008002,
    0x8000000000000080,0x8000000000000080,0x8000000000000080,0x8000000000000080, 0x8000000000000080,0x8000000000000080,0x8000000000000080,0x8000000000000080,
    0x000000000000800A,0x000000000000800A,0x000000000000800A,0x000000000000800A, 0x000000000000800A,0x000000000000800A,0x000000000000800A,0x000000000000800A,
    0x800000008000000A,0x800000008000000A,0x800000008000000A,0x800000008000000A, 0x800000008000000A,0x800000008000000A,0x800000008000000A,0x800000008000000A,
    0x8000000080008081,0x8000000080008081,0x8000000080008081,0x8000000080008081, 0x8000000080008081,0x8000000080008081,0x8000000080008081,0x8000000080008081,
    0x8000000000008080,0x8000000000008080,0x8000000000008080,0x8000000000008080, 0x8000000000008080,0x8000000000008080,0x8000000000008080,0x8000000000008080,
    0x0000000080000001,0x0000000080000001,0x0000000080000001,0x0000000080000001, 0x0000000080000001,0x0000000080000001,0x0000000080000001,0x0000000080000001,
    0x8000000080008008,0x8000000080008008,0x8000000080008008,0x8000000080008008, 0x8000000080008008,0x8000000080008008,0x8000000080008008,0x8000000080008008
  };

  const __m512i l0 = _mm512_set_epi64(11,10,3,2,9,8,1,0);
  const __m512i l1 = _mm512_set_epi64(15,14,7,6,13,12,5,4);
  const __m512i l2 = _mm512_set_epi64(11,10,9,8,3,2,1,0);
  const __m512i l3 = _mm512_set_epi64(15,14,13,12,7,6,5,4);

  ALIGN uint8_t temP[8][200];
  int j;
  __m512i s[25], t[8],r[8];
  __m128i auxl[6];
  __m256i auxM;
  char* in[8];
  int rt;
  
  for(i=0;i<8;i++){
    in[i]=in_e[i];
  }

  int rsiz = 200-2*ra;
  rt = (rsiz/64);
  init_S_zeros(j);
  for(;inlen >=rsiz; inlen -=rsiz){
	  loadM(j,rt,in);
		  keccakf(s); 

	  for(i=0;i<8;i++){
	      in[i] += rsiz;
    	  }
  }
/*   for(i=0;i<25;i++)
		  print_512(s[i]);
	  getchar();
*/
  for(i=0;i<8;i++){
    memset(temP[i], 0, 168*sizeof(uint8_t));
    memcpy(temP[i], in[i], inlen);
    temP[i][inlen] = 0x06;
    temP[i][rsiz - 1] |= 0x80;
  }
  loadM(j,rt,temP); 
  keccakf(s);

  back(j,(rt*2)+1,temP);

  if(rsiz < out){
  	for(i=0;i<8;i++)
		memcpy(md[i], temP[i], rsiz);
  	out -= rsiz;
  	back(j,(rt*2)+1,temP);
  	keccakf(s);
  }
  for(i=0;i<8;i++)
		memcpy(md[i], temP[i], rsiz);

  return 0;
}


