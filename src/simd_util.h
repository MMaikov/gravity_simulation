#ifndef SIMD_UTIL_H
#define SIMD_UTIL_H

#include <immintrin.h>

#if defined(__AVX512F__)
__m512i avx512_clamp_epi32(__m512i val, __m512i min, __m512i max);
#endif

#if defined(__AVX2__)
__m128i mm256_cvtepi32_epi8_avx2(__m256i input);
#endif

#endif //SIMD_UTIL_H
