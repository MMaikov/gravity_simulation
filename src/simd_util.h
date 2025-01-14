#ifndef SIMD_UTIL_H
#define SIMD_UTIL_H

#include <immintrin.h>

// https://stackoverflow.com/a/35270026
float hsum_ps_sse3(__m128 v);
float hsum256_ps_avx(__m256 v);

// Inspired by the stackoverflow answer above
float hsum_ps_sse(__m128 v);

__m512i avx512_clamp_epi32(__m512i val, __m512i min, __m512i max);

#endif //SIMD_UTIL_H
