#include "simd_util.h"

// https://stackoverflow.com/a/35270026
float hsum_ps_sse3(__m128 v) {
    __m128 shuf = _mm_movehdup_ps(v);        // broadcast elements 3,1 to 2,0
    __m128 sums = _mm_add_ps(v, shuf);
    shuf        = _mm_movehl_ps(shuf, sums); // high half -> low half
    sums        = _mm_add_ss(sums, shuf);
    return        _mm_cvtss_f32(sums);
}

// https://stackoverflow.com/a/35270026
float hsum256_ps_avx(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1); // high 128
    vlow  = _mm_add_ps(vlow, vhigh);     // add the low 128
    return hsum_ps_sse3(vlow);         // and inline the sse3 version, which is optimal for AVX
    // (no wasted instructions, and all of them are the 4B minimum)
}

// Inspired by the stackoverflow answer above
float hsum_ps_sse(__m128 v) {
    __m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 1, 2, 0)); // Shuffle elements 3,1 to 2,0
    __m128 sums = _mm_add_ps(v, shuf);
    shuf        = _mm_movehl_ps(shuf, sums); // high half -> low half
    sums        = _mm_add_ss(sums, shuf);
    return        _mm_cvtss_f32(sums);
}

__m512i avx512_clamp_epi32(__m512i val, __m512i min, __m512i max) {
    return _mm512_max_epi32(min, _mm512_min_epi32(val, max));
}

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