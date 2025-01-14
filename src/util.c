#include "util.h"

#include <immintrin.h>
#include "simd_util.h"

#include "config.h"

#include "stb_image_write.h"

void set_pixel(SDL_Surface* surface, uint32_t x, uint32_t y, Uint32 pixel)
{
    Uint8* target_pixel = (Uint8*)surface->pixels + y * surface->pitch + x * 4;
    *(Uint32*)target_pixel = pixel;
}

void blur5x5(float* values, uint32_t width, uint32_t height, float* tmp_buf)
{
    const float multiplier = 1/16.0f;

#if USE_SIMD
#if defined(__AVX512F__) && USE_AVX512
    const __m512 multiplier_f = _mm512_set1_ps(multiplier);
    const __m512 four = _mm512_set1_ps(4.0f);
    const __m512 six = _mm512_set1_ps(6.0f);
    for (uint32_t y = 2; y < height-2; y++) {
        uint32_t x;
        for (x = 0; x < width-15; x += AVX512_FLOATS) {
            const uint32_t index = x+y*width;
            const __m512 values_2pw = _mm512_loadu_ps(values + index + width * 2);
            const __m512 values_pw = _mm512_loadu_ps(values + index + width);
            const __m512 values_f = _mm512_loadu_ps(values + index);
            const __m512 values_nw = _mm512_loadu_ps(values + index - width);
            const __m512 values_2nw = _mm512_loadu_ps(values + index - 2*width);

            __m512 val = _mm512_fmadd_ps(values_pw, four, values_2pw);
            val = _mm512_fmadd_ps(values_f, six, val);
            val = _mm512_fmadd_ps(values_nw, four, val);
            val = _mm512_add_ps(val, values_2nw);
            val = _mm512_mul_ps(val, multiplier_f);

            _mm512_storeu_ps(tmp_buf + index, val);
        }
        for (; x < width; x++){
            const uint32_t index = x+y*width;
            tmp_buf[index] = 0;
            tmp_buf[index] += values[index + width * 2];
            tmp_buf[index] += values[  index + width  ] * 4;
            tmp_buf[index] += values[      index      ] * 6;
            tmp_buf[index] += values[  index - width  ] * 4;
            tmp_buf[index] += values[index - width * 2];
            tmp_buf[index] *= multiplier;
        }
    }


    for (uint32_t y = 0; y < height; y++) {
        uint32_t x;
        for (x = 2; x < width-2-15; x += AVX512_FLOATS) {
            const uint32_t index = x+y*width;

            const __m512 tmp_buf_p2 = _mm512_loadu_ps(tmp_buf + index + 2);
            const __m512 tmp_buf_p1 = _mm512_loadu_ps(tmp_buf + index + 1);
            const __m512 tmp_buf_f = _mm512_loadu_ps(tmp_buf + index);
            const __m512 tmp_buf_n1 = _mm512_loadu_ps(tmp_buf + index - 1);
            const __m512 tmp_buf_n2 = _mm512_loadu_ps(tmp_buf + index - 2);

            __m512 val = _mm512_fmadd_ps(tmp_buf_p1, four, tmp_buf_p2);
            val = _mm512_fmadd_ps(tmp_buf_f, six, val);
            val = _mm512_fmadd_ps(tmp_buf_n1, four, val);
            val = _mm512_add_ps(val, tmp_buf_n2);
            val = _mm512_mul_ps(val, multiplier_f);

            _mm512_storeu_ps(values + index, val);
        }

        for (; x < width-2; x++) {
            const uint32_t index = x+y*width;
            values[index] = 0;
            values[index] += tmp_buf[index + 2];
            values[index] += tmp_buf[index + 1] * 4;
            values[index] += tmp_buf[  index  ] * 6;
            values[index] += tmp_buf[index - 1] * 4;
            values[index] += tmp_buf[index - 2];
            values[index] *= multiplier;
        }
    }
#elif defined(__AVX__) && USE_AVX
    const __m256 multiplier_f = _mm256_set1_ps(multiplier);
    const __m256 four = _mm256_set1_ps(4.0f);
    const __m256 six = _mm256_set1_ps(6.0f);
    for (uint32_t y = 2; y < height-2; y++) {
        uint32_t x;
        for (x = 0; x < width-7; x += AVX_FLOATS) {
            const uint32_t index = x+y*width;
            const __m256 values_2pw = _mm256_loadu_ps(values + index + width * 2);
            const __m256 values_pw = _mm256_loadu_ps(values + index + width);
            const __m256 values_f = _mm256_loadu_ps(values + index);
            const __m256 values_nw = _mm256_loadu_ps(values + index - width);
            const __m256 values_2nw = _mm256_loadu_ps(values + index - 2*width);

#if USE_FMA
            __m256 val = _mm256_fmadd_ps(values_pw, four, values_2pw);
            val = _mm256_fmadd_ps(values_f, six, val);
            val = _mm256_fmadd_ps(values_nw, four, val);
#else
            __m256 val = _mm256_add_ps(_mm256_mul_ps(values_pw, four), values_2pw);
            val = _mm256_add_ps(_mm256_mul_ps(values_f, six), val);
            val = _mm256_add_ps(_mm256_mul_ps(values_nw, four), val);
#endif

            val = _mm256_add_ps(val, values_2nw);
            val = _mm256_mul_ps(val, multiplier_f);

            _mm256_storeu_ps(tmp_buf + index, val);
        }
        for (; x < width; x++){
            const uint32_t index = x+y*width;
            tmp_buf[index] = 0;
            tmp_buf[index] += values[index + width * 2];
            tmp_buf[index] += values[  index + width  ] * 4;
            tmp_buf[index] += values[      index      ] * 6;
            tmp_buf[index] += values[  index - width  ] * 4;
            tmp_buf[index] += values[index - width * 2];
            tmp_buf[index] *= multiplier;
        }
    }


    for (uint32_t y = 0; y < height; y++) {
        uint32_t x;
        for (x = 2; x < width-2-7; x += AVX_FLOATS) {
            const uint32_t index = x+y*width;

            const __m256 tmp_buf_p2 = _mm256_loadu_ps(tmp_buf + index + 2);
            const __m256 tmp_buf_p1 = _mm256_loadu_ps(tmp_buf + index + 1);
            const __m256 tmp_buf_f = _mm256_loadu_ps(tmp_buf + index);
            const __m256 tmp_buf_n1 = _mm256_loadu_ps(tmp_buf + index - 1);
            const __m256 tmp_buf_n2 = _mm256_loadu_ps(tmp_buf + index - 2);

#if USE_FMA
            __m256 val = _mm256_fmadd_ps(tmp_buf_p1, four, tmp_buf_p2);
            val = _mm256_fmadd_ps(tmp_buf_f, six, val);
            val = _mm256_fmadd_ps(tmp_buf_n1, four, val);
#else
            __m256 val = _mm256_add_ps(_mm256_mul_ps(tmp_buf_p1, four), tmp_buf_p2);
            val = _mm256_add_ps(_mm256_mul_ps(tmp_buf_f, six), val);
            val = _mm256_add_ps(_mm256_mul_ps(tmp_buf_n1, four), val);
#endif

            val = _mm256_add_ps(val, tmp_buf_n2);
            val = _mm256_mul_ps(val, multiplier_f);

            _mm256_storeu_ps(values + index, val);
        }

        for (; x < width-2; x++) {
            const uint32_t index = x+y*width;
            values[index] = 0;
            values[index] += tmp_buf[index + 2];
            values[index] += tmp_buf[index + 1] * 4;
            values[index] += tmp_buf[  index  ] * 6;
            values[index] += tmp_buf[index - 1] * 4;
            values[index] += tmp_buf[index - 2];
            values[index] *= multiplier;
        }
    }
#elif defined(__SSE__)
    const __m128 multiplier_f = _mm_set1_ps(multiplier);
    const __m128 four = _mm_set1_ps(4.0f);
    const __m128 six = _mm_set1_ps(6.0f);
    for (uint32_t y = 2; y < height-2; y++) {
        uint32_t x;
        for (x = 0; x < width-3; x += SSE_FLOATS) {
            const uint32_t index = x+y*width;
            const __m128 values_2pw = _mm_loadu_ps(values + index + width * 2);
            const __m128 values_pw = _mm_loadu_ps(values + index + width);
            const __m128 values_f = _mm_loadu_ps(values + index);
            const __m128 values_nw = _mm_loadu_ps(values + index - width);
            const __m128 values_2nw = _mm_loadu_ps(values + index - 2*width);

#if USE_FMA
            __m128 val = _mm_fmadd_ps(values_pw, four, values_2pw);
            val = _mm_fmadd_ps(values_f, six, val);
            val = _mm_fmadd_ps(values_nw, four, val);
#else
            __m128 val = _mm_add_ps(_mm_mul_ps(values_pw, four), values_2pw);
            val = _mm_add_ps(_mm_mul_ps(values_f, six), val);
            val = _mm_add_ps(_mm_mul_ps(values_nw, four), val);
#endif

            val = _mm_add_ps(val, values_2nw);
            val = _mm_mul_ps(val, multiplier_f);

            _mm_storeu_ps(tmp_buf + index, val);
        }
        for (; x < width; x++){
            const uint32_t index = x+y*width;
            tmp_buf[index] = 0;
            tmp_buf[index] += values[index + width * 2];
            tmp_buf[index] += values[  index + width  ] * 4;
            tmp_buf[index] += values[      index      ] * 6;
            tmp_buf[index] += values[  index - width  ] * 4;
            tmp_buf[index] += values[index - width * 2];
            tmp_buf[index] *= multiplier;
        }
    }


    for (uint32_t y = 0; y < height; y++) {
        uint32_t x;
        for (x = 2; x < width-2-3; x += SSE_FLOATS) {
            const uint32_t index = x+y*width;

            const __m128 tmp_buf_p2 = _mm_loadu_ps(tmp_buf + index + 2);
            const __m128 tmp_buf_p1 = _mm_loadu_ps(tmp_buf + index + 1);
            const __m128 tmp_buf_f = _mm_loadu_ps(tmp_buf + index);
            const __m128 tmp_buf_n1 = _mm_loadu_ps(tmp_buf + index - 1);
            const __m128 tmp_buf_n2 = _mm_loadu_ps(tmp_buf + index - 2);

#if USE_FMA
            __m128 val = _mm_fmadd_ps(tmp_buf_p1, four, tmp_buf_p2);
            val = _mm_fmadd_ps(tmp_buf_f, six, val);
            val = _mm_fmadd_ps(tmp_buf_n1, four, val);
#else
            __m128 val = _mm_add_ps(_mm_mul_ps(tmp_buf_p1, four), tmp_buf_p2);
            val = _mm_add_ps(_mm_mul_ps(tmp_buf_f, six), val);
            val = _mm_add_ps(_mm_mul_ps(tmp_buf_n1, four), val);
#endif

            val = _mm_add_ps(val, tmp_buf_n2);
            val = _mm_mul_ps(val, multiplier_f);

            _mm_storeu_ps(values + index, val);
        }

        for (; x < width-2; x++) {
            const uint32_t index = x+y*width;
            values[index] = 0;
            values[index] += tmp_buf[index + 2];
            values[index] += tmp_buf[index + 1] * 4;
            values[index] += tmp_buf[  index  ] * 6;
            values[index] += tmp_buf[index - 1] * 4;
            values[index] += tmp_buf[index - 2];
            values[index] *= multiplier;
        }
    }
#else
#error SIMD not supported
#endif
#else
    for (uint32_t y = 2; y < height-2; y++){
        for (uint32_t x = 0; x < width; x++){
            const uint32_t index = x + y * width;
            tmp_buf[index] = 0;
            tmp_buf[index] += values[index + width * 2];
            tmp_buf[index] += values[  index + width  ] * 4;
            tmp_buf[index] += values[      index      ] * 6;
            tmp_buf[index] += values[  index - width  ] * 4;
            tmp_buf[index] += values[index - width * 2];
            tmp_buf[index] *= multiplier;
        }
    }
    for (uint32_t y = 0; y < height; y++){
        for (uint32_t x = 2; x < width-2; x++){
            const uint32_t index = x + y * width;
            values[index] = 0;
            values[index] += tmp_buf[index + 2];
            values[index] += tmp_buf[index + 1] * 4;
            values[index] += tmp_buf[  index  ] * 6;
            values[index] += tmp_buf[index - 1] * 4;
            values[index] += tmp_buf[index - 2];
            values[index] *= multiplier;
        }
    }
#endif
}

bool save_image(uint8_t* pixels, uint32_t width, uint32_t height, uint32_t name_index) {

    if (!SDL_CreateDirectory("img/")) {
        SDL_Log("Failed to create directory\n%s\n", SDL_GetError());
        return false;
    }

    char filename[32];
    (void) SDL_snprintf(filename, COUNT_OF(filename), "img/%07d.jpg", name_index);

    return stbi_write_jpg(filename, width, height, 1, pixels, 90);
}