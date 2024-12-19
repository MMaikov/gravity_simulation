#ifndef SIMD_UTIL_H
#define SIMD_UTIL_H

#include <immintrin.h>

// https://stackoverflow.com/a/35270026
float hsum_ps_sse3(__m128 v);
float hsum256_ps_avx(__m256 v);

#endif //SIMD_UTIL_H
