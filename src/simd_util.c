#include "simd_util.h"

#if defined(__AVX512F__)
__m512i avx512_clamp_epi32(__m512i val, __m512i min, __m512i max) {
    return _mm512_max_epi32(min, _mm512_min_epi32(val, max));
}
#endif

#if defined(__AVX2__)
// ChatGPT
__m128i mm256_cvtepi32_epi8_avx2(__m256i input) {
    // Constants for saturation
    __m256i min_val = _mm256_set1_epi32(-128); // Minimum value for int8_t
    __m256i max_val = _mm256_set1_epi32(127);  // Maximum value for int8_t

    // Saturate values
    __m256i clamped = _mm256_max_epi32(_mm256_min_epi32(input, max_val), min_val);

    // Extract the lower and upper 128-bit lanes
    __m128i lo = _mm256_castsi256_si128(clamped);
    __m128i hi = _mm256_extracti128_si256(clamped, 1);

    // Pack 32-bit integers to 16-bit integers (saturation)
    __m128i packed16 = _mm_packs_epi32(lo, hi);

    // Pack 16-bit integers to 8-bit integers (saturation)
    __m128i packed8 = _mm_packs_epi16(packed16, _mm_setzero_si128());

    return packed8;
}
#endif