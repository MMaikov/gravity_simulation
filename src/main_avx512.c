#include "main_avx512.h"

#include "simd_util.h"

#include "util.h"

void write_to_window_buffer_avx512(float* window_values, struct particle_system* particle_system, float view_pos_x, float view_pos_y, float view_scale)
{
    const __m512 view_scale_f = _mm512_set1_ps(view_scale);
    const __m512 view_pos_x_f = _mm512_set1_ps(view_pos_x);
    const __m512 view_pos_y_f = _mm512_set1_ps(view_pos_y);
    const float* pos_x = particle_system->pos_x;
    const float* pos_y = particle_system->pos_y;

    size_t i;
    for (i = 0; i < particle_system->num_particles - 15; i += AVX512_FLOATS) {
        const __m512 pos_x_f = _mm512_load_ps(pos_x + i);
        const __m512 pos_y_f = _mm512_load_ps(pos_y + i);

        __m512i x = _mm512_cvtps_epi32(_mm512_mul_ps(view_scale_f, _mm512_sub_ps(pos_x_f, view_pos_x_f)));
        __m512i y = _mm512_cvtps_epi32(_mm512_mul_ps(view_scale_f, _mm512_sub_ps(pos_y_f, view_pos_y_f)));

        x = _mm512_add_epi32(x, _mm512_set1_epi32(WINDOW_WIDTH/2));
        y = _mm512_add_epi32(y, _mm512_set1_epi32(WINDOW_HEIGHT/2));

        x = avx512_clamp_epi32(x, _mm512_setzero_epi32(), _mm512_set1_epi32(WINDOW_WIDTH-1));
        y = avx512_clamp_epi32(y, _mm512_setzero_epi32(), _mm512_set1_epi32(WINDOW_HEIGHT-1));

        const __m512i index = _mm512_add_epi32(x, _mm512_mullo_epi32(y, _mm512_set1_epi32(WINDOW_WIDTH)));

#if AVX512_AVOID_GATHERSCATTER
        uint32_t index_32[AVX512_FLOATS];
        _mm512_storeu_si512(index_32, index);
        for (int k = 0; k < AVX512_FLOATS; k++) {
            window_values[index_32[k]] += 1.0f;
        }
#else
        __m512 window_values_f = _mm512_i32gather_ps(index, window_values, 4);
        window_values_f = _mm512_add_ps(window_values_f, _mm512_set1_ps(1.0f));
        _mm512_i32scatter_ps(window_values, index, window_values_f, 4);
#endif
    }

    for (; i < particle_system->num_particles; i++) {
        int32_t x = (int32_t)((particle_system->pos_x[i] - view_pos_x)*view_scale);
        int32_t y = (int32_t)((particle_system->pos_y[i] - view_pos_y)*view_scale);
        x = SDL_clamp(x + WINDOW_WIDTH/2, 0, WINDOW_WIDTH-1);
        y = SDL_clamp(y + WINDOW_HEIGHT/2, 0, WINDOW_HEIGHT-1);
        window_values[x + y*WINDOW_WIDTH] += 1.0f;
    }
}

void write_to_surface_avx512(SDL_Surface* surface, float view_scale, float brightness, float* window_values, uint8_t* window_chars)
{
    const float view_brightness = view_scale*brightness;
    const __m512 view_brightness_f = _mm512_set1_ps(view_brightness);
    size_t i;
    for (i = 0; i < WINDOW_WIDTH*WINDOW_HEIGHT-16; i += AVX512_FLOATS) {
        __m512i values_i = _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(window_values + i), view_brightness_f));
        _mm512_storeu_ps(window_values + i, _mm512_setzero_ps());

        values_i = _mm512_min_epi32(values_i, _mm512_set1_epi32(255));

        _mm512_storeu_epi32(surface->pixels + i*4, _mm512_mullo_epi32(_mm512_set1_epi32(0x01010101u), values_i));

        _mm_storeu_si128((__m128i_u*)(window_chars + i), _mm512_cvtepi32_epi8(values_i));
    }

    for (; i < WINDOW_WIDTH*WINDOW_HEIGHT-1; i++) {
        int value = (int)(window_values[i] * view_brightness);
        window_values[i] = 0.0f;
        value = SDL_min(value, 255);
        set_pixel(surface, i, 0, 0x01010101u * value);
        window_chars[i] = (uint8_t)value;
    }
}
