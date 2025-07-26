#include "main_avx2.h"

#include "config.h"
#include "simd_util.h"
#include "util.h"

void write_to_window_buffer_avx2(float* window_values, struct particle_system* particle_system, float view_pos_x, float view_pos_y, float view_scale)
{
    const __m256 view_scale_f = _mm256_set1_ps(view_scale);
    const __m256 view_pos_x_f = _mm256_set1_ps(view_pos_x);
    const __m256 view_pos_y_f = _mm256_set1_ps(view_pos_y);
    const float* pos_x = particle_system->pos_x;
    const float* pos_y = particle_system->pos_y;

    size_t i = 0;
    for (; i < particle_system->num_particles - AVX_FLOATS - 1; i += AVX_FLOATS) {

        const __m256 pos_x_f = _mm256_load_ps(pos_x + i);
        const __m256 pos_y_f = _mm256_load_ps(pos_y + i);

        __m256i x = _mm256_cvtps_epi32(_mm256_mul_ps(view_scale_f, _mm256_sub_ps(pos_x_f, view_pos_x_f)));
        __m256i y = _mm256_cvtps_epi32(_mm256_mul_ps(view_scale_f, _mm256_sub_ps(pos_y_f, view_pos_y_f)));

        x = _mm256_add_epi32(x, _mm256_set1_epi32(WINDOW_WIDTH/2));
        y = _mm256_add_epi32(y, _mm256_set1_epi32(WINDOW_HEIGHT/2));

        x = avx256_clamp_epi32(x, _mm256_setzero_si256(), _mm256_set1_epi32(WINDOW_WIDTH-1));
        y = avx256_clamp_epi32(y, _mm256_setzero_si256(), _mm256_set1_epi32(WINDOW_HEIGHT-1));

        const __m256i index = _mm256_add_epi32(x, _mm256_mullo_epi32(y, _mm256_set1_epi32(WINDOW_WIDTH)));

        uint32_t index_32[AVX_FLOATS];
        _mm256_storeu_si256(index_32, index);

        for (int k = 0; k < AVX_FLOATS; k++) {
            window_values[index_32[k]] += 1.0f;
        }
    }

    for (; i < particle_system->num_particles; i++) {
        int32_t x = (int32_t)((particle_system->pos_x[i] - view_pos_x)*view_scale);
        int32_t y = (int32_t)((particle_system->pos_y[i] - view_pos_y)*view_scale);
        x = SDL_clamp(x + WINDOW_WIDTH/2, 0, WINDOW_WIDTH-1);
        y = SDL_clamp(y + WINDOW_HEIGHT/2, 0, WINDOW_HEIGHT-1);
        window_values[x + y*WINDOW_WIDTH] += 1.0f;
    }
}

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