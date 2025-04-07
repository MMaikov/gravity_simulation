#include "main_avx2.h"

#include "config.h"
#include "simd_util.h"
#include "util.h"

void write_to_surface_avx2(SDL_Surface* surface, float view_scale, float brightness, float* window_values, uint8_t* window_chars)
{
    const float view_brightness = view_scale*brightness;
    const __m256 view_brightness_f = _mm256_set1_ps(view_brightness);
    size_t i;
    for (i = 0; i < WINDOW_WIDTH*WINDOW_HEIGHT-8; i += AVX_FLOATS) {
        __m256i values_i = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(window_values + i), view_brightness_f));
        _mm256_storeu_ps(window_values + i, _mm256_setzero_ps());

        values_i = _mm256_min_epi32(values_i, _mm256_set1_epi32(255));

        _mm256_storeu_si256(surface->pixels + i*4, _mm256_mullo_epi32(_mm256_set1_epi32(0x01010101u), values_i));

        _mm_storeu_si64(window_chars + i, mm256_cvtepi32_epi8_avx2(values_i));
    }
    for (; i < WINDOW_WIDTH*WINDOW_HEIGHT-1; i++) {
        int value = (int)(window_values[i] * view_brightness);
        window_values[i] = 0.0f;
        value = SDL_min(value, 255);
        set_pixel(surface, i, 0, 0x01010101u * value);
        window_chars[i] = (uint8_t)value;
    }
}