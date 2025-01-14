#include <SDL.h>

#include "config.h"
#include "particle_system.h"
#include "timer.h"
#include "util.h"

#include "simd_util.h"

static void write_to_window_buffer(float* window_values, struct particle_system* particle_system, float view_pos_x, float view_pos_y, float view_scale) {

#if USE_SIMD && defined(__AVX512F__) && USE_AVX512

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
            __m512 window_values_f = _mm512_i32gather_ps(index, window_values, 4);
            window_values_f = _mm512_add_ps(window_values_f, _mm512_set1_ps(1.0f));
            _mm512_i32scatter_ps(window_values, index, window_values_f, 4);
        }

        for (; i < particle_system->num_particles; i++) {
            int32_t x = (int32_t)((particle_system->pos_x[i] - view_pos_x)*view_scale);
            int32_t y = (int32_t)((particle_system->pos_y[i] - view_pos_y)*view_scale);
            x = SDL_clamp(x + WINDOW_WIDTH/2, 0, WINDOW_WIDTH-1);
            y = SDL_clamp(y + WINDOW_HEIGHT/2, 0, WINDOW_HEIGHT-1);
            window_values[x + y*WINDOW_WIDTH] += 1.0f;
        }
#else
        for (size_t i = 0; i < particle_system->num_particles; i++) {
            int32_t x = (int32_t)((particle_system->pos_x[i] - view_pos_x)*view_scale);
            int32_t y = (int32_t)((particle_system->pos_y[i] - view_pos_y)*view_scale);
            x = SDL_clamp(x + WINDOW_WIDTH/2, 0, WINDOW_WIDTH-1);
            y = SDL_clamp(y + WINDOW_HEIGHT/2, 0, WINDOW_HEIGHT-1);
            window_values[x + y*WINDOW_WIDTH] += 1.0f;
        }
#endif
}

static void write_to_surface(SDL_Surface* surface, float view_scale, float brightness, float* window_values, uint8_t* window_chars) {
#if USE_SIMD
#if defined(__AVX512F__) && USE_AVX512
    const float view_brightness = view_scale*brightness;
    const __m512 view_brightness_f = _mm512_set1_ps(view_brightness);
    size_t i;
    for (i = 0; i < WINDOW_WIDTH*WINDOW_HEIGHT-16; i += AVX512_FLOATS) {
        __m512i values_i = _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(window_values + i), view_brightness_f));
        _mm512_storeu_ps(window_values + i, _mm512_setzero_ps());

        values_i = _mm512_min_epi32(values_i, _mm512_set1_epi32(255));

        _mm512_storeu_epi32(surface->pixels + i*4, _mm512_mullo_epi32(_mm512_set1_epi32(0x01010101u), values_i));

        _mm_storeu_epi8(window_chars + i, _mm512_cvtepi32_epi8(values_i));
    }
#elif defined(__AVX2__) && USE_AVX
    const float view_brightness = view_scale*brightness;
    const __m256 view_brightness_f = _mm256_set1_ps(view_brightness);
    size_t i;
    for (i = 0; i < WINDOW_WIDTH*WINDOW_HEIGHT-8; i += AVX_FLOATS) {
        __m256i values_i = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(window_values + i), view_brightness_f));
        _mm256_storeu_ps(window_values + i, _mm256_setzero_ps());

        values_i = _mm256_min_epi32(values_i, _mm256_set1_epi32(255));

        _mm256_storeu_epi32(surface->pixels + i*4, _mm256_mullo_epi32(_mm256_set1_epi32(0x01010101u), values_i));

        _mm_storeu_epi8(window_chars + i, _mm256_cvtepi32_epi8(values_i));
    }
#elif defined(__SSE2__)
    const float view_brightness = view_scale*brightness;
    const __m128 view_brightness_f = _mm_set1_ps(view_brightness);
    size_t i;
    for (i = 0; i < WINDOW_WIDTH*WINDOW_HEIGHT-4; i += SSE_FLOATS) {
        __m128i values_i = _mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(window_values + i), view_brightness_f));
        _mm_storeu_ps(window_values + i, _mm_setzero_ps());

        values_i = _mm_min_epi32(values_i, _mm_set1_epi32(255));

        _mm_storeu_epi32(surface->pixels + i*4, _mm_mullo_epi32(_mm_set1_epi32(0x01010101u), values_i));

        _mm_storeu_epi8(window_chars + i, _mm_cvtepi32_epi8(values_i));
    }
#else
#error SIMD not supported
#endif
    for (; i < WINDOW_WIDTH*WINDOW_HEIGHT-1; i++) {
        int value = (int)(window_values[i] * view_brightness);
        window_values[i] = 0.0f;
        value = SDL_min(value, 255);
        set_pixel(surface, i, 0, 0x01010101u * value);
        window_chars[i] = (uint8_t)value;
    }
#else
    const float view_brightness = view_scale*brightness;
    for (size_t i = 0; i < WINDOW_WIDTH*WINDOW_HEIGHT-1; i++) {
        int value = (int)(window_values[i] * view_brightness);
        window_values[i] = 0.0f;
        value = SDL_min(value, 255);
        set_pixel(surface, i, 0, 0x01010101u * value);
        window_chars[i] = (uint8_t)value;
    }
#endif
}

int main(const int argc, char** argv)
{
    uint32_t particle_count = NUM_PARTICLES;

    if (argc > 1) {
        if (SDL_strncmp("--particles", argv[1], SDL_strlen(argv[1])) == 0) {
            if (argc > 2) {
                particle_count = SDL_strtoul(argv[2], NULL, 10);
            } else {
                SDL_Log("No particle count specified");
            }
        }
    }

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to initialize SDL!\n%s\n", SDL_GetError());
        goto cleanup1;
    }

    SDL_Window* window = SDL_CreateWindow(WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT, 0);
    if (window == NULL) {
        SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to create SDL Window!\n%s\n", SDL_GetError());
        goto cleanup2;
    }

    struct particle_system particle_system = { 0 };
    if (!particle_system_init(&particle_system, particle_count)) {
        goto cleanup3;
    }

    SDL_Log("Initialized particle system with %d particles", particle_system.num_particles);
    SDL_Log("Number of threads: %d", particle_system.num_threads);

    SDL_Surface* surface = SDL_GetWindowSurface(window);
    if (surface == NULL) {
        SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to get surface!\n%s\n", SDL_GetError());
        goto cleanup4;
    }

    float* window_values = SDL_calloc(WINDOW_WIDTH * WINDOW_HEIGHT, sizeof(*window_values));
    float* tmp_buf = SDL_calloc(WINDOW_WIDTH * WINDOW_HEIGHT, sizeof(*tmp_buf));
    uint8_t* window_chars = SDL_calloc(WINDOW_WIDTH * WINDOW_HEIGHT, sizeof(*window_chars));

    if (window_values == NULL || tmp_buf == NULL || window_chars == NULL) {
        SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to allocate memory!\n%s\n", SDL_GetError());
        goto cleanup4;
    }

    const SDL_Rect window_rectangle = { .x = 0, .y = 0, .w = WINDOW_WIDTH, .h = WINDOW_HEIGHT };
    const SDL_PixelFormatDetails* pixel_format_details = SDL_GetPixelFormatDetails(surface->format);
    if (pixel_format_details == NULL) {
        SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to get pixel format details!\n%s\n", SDL_GetError());
        goto cleanup5;
    }

    float view_pos_x = 0;
    float view_pos_y = 0;
    float view_scale = WINDOW_HEIGHT / 2.0f;
    float brightness = 10000.0f / (float)particle_system.num_particles;

    struct timer update_timer = {0};
    struct timer render_timer = {0};

    float mouse_pos_x = 0.0f;
    float mouse_pos_y = 0.0f;

    bool simulate = false;
    bool record = false;

    int32_t num_updates = 1;
    uint32_t img_num = 0;
    float dt = 1e-6f;

    SDL_Event e;
    bool running = true;
    while (running) {
        SDL_GetMouseState(&mouse_pos_x, &mouse_pos_y);
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_EVENT_QUIT) {
                running = false;
            } else if (e.type == SDL_EVENT_MOUSE_WHEEL) {
                const float zoom_factor = (e.wheel.y < 0) ? 0.95f : 1.05f;

                float change_x = mouse_pos_x - WINDOW_WIDTH / 2.0f;
                float change_y = mouse_pos_y - WINDOW_HEIGHT / 2.0f;
                view_pos_x = view_pos_x + change_x / view_scale;
                view_pos_y = view_pos_y + change_y / view_scale;
                view_scale *= zoom_factor;
                change_x = mouse_pos_x - WINDOW_WIDTH / 2.0f;
                change_y = mouse_pos_y - WINDOW_HEIGHT / 2.0f;
                view_pos_x = view_pos_x - change_x / view_scale;
                view_pos_y = view_pos_y - change_y / view_scale;
            } else if (e.type == SDL_EVENT_KEY_DOWN) {
                switch (e.key.key) {
                    case SDLK_ESCAPE:
                        running = false;
                        break;
                    case SDLK_SPACE:
                        simulate ^= true;
                        break;
                    case SDLK_LEFT:
                        view_pos_x -= WINDOW_HEIGHT / view_scale * 0.01f;
                        break;
                    case SDLK_RIGHT:
                        view_pos_x += WINDOW_WIDTH / view_scale * 0.01f;
                        break;
                    case SDLK_UP:
                        view_pos_y -= WINDOW_HEIGHT / view_scale * 0.01f;
                        break;
                    case SDLK_DOWN:
                        view_pos_y += WINDOW_HEIGHT / view_scale * 0.01f;
                        break;
                    case SDLK_R:
                        particle_system_reset(&particle_system);
                        timer_reset(&update_timer);
                        simulate = false;
                        break;
                    case SDLK_S:
                        simulate = true;
                        record = true;
                        break;
                    case SDLK_O:
                        view_scale *= 0.98f;
                        break;
                    case SDLK_I:
                        view_scale *= 1.02f;
                        break;
                    case SDLK_F:
                        num_updates += 1;
                        break;
                    case SDLK_G:
                        num_updates -= 1;
                        num_updates = SDL_max(num_updates, 0);
                        break;
                    case SDLK_V:
                        dt *= 1.1f;
                        break;
                    case SDLK_B:
                        dt *= 1.0f / 1.1f;
                        break;
                    case SDLK_E:
                        brightness *= 1.05f;
                        break;
                    case SDLK_W:
                        brightness *= 0.95f;
                        break;

                    default: break;
                }
            }
        }

        if (simulate) {
            timer_start(&update_timer);
            for (size_t i = 0; i < num_updates; i++) {
                particle_system_update(&particle_system, dt);
            }
            timer_stop(&update_timer);
        } else {
            timer_reset(&update_timer);
        }

        timer_start(&render_timer);

        if (!SDL_FillSurfaceRect(surface, &window_rectangle, SDL_MapRGB(pixel_format_details, NULL, 0, 0, 0))) {
            SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to fill surface!\n%s\n", SDL_GetError());
            goto cleanup5;
        }

        write_to_window_buffer(window_values, &particle_system, view_pos_x, view_pos_y, view_scale);

        blur5x5(window_values, WINDOW_WIDTH, WINDOW_HEIGHT, tmp_buf);

        write_to_surface(surface, view_scale, brightness, window_values, window_chars);

        timer_stop(&render_timer);

        char update_time_buf[10] = {0};
        timer_elapsed_str(&update_timer, COUNT_OF(update_time_buf), update_time_buf);
        char render_time_buf[10] = {0};
        timer_elapsed_str(&render_timer, COUNT_OF(render_time_buf), render_time_buf);
        char title_buf[120] = {0};
        SDL_snprintf(title_buf, COUNT_OF(title_buf), "%s - update: %s render: %s updates: %d dt: %.10e", WINDOW_TITLE, update_time_buf, render_time_buf, num_updates, dt);

        if (!SDL_SetWindowTitle(window, title_buf)) {
            SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to set window title!\n%s\n", SDL_GetError());
            goto cleanup5;
        }

        if (record) {
            if (!save_image(window_chars, WINDOW_WIDTH, WINDOW_HEIGHT, img_num)) {
                SDL_Log("Failed to save image!");
                record = false;
            }
            img_num++;
        }

        if (!SDL_UpdateWindowSurface(window)) {
            SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to update window surface!\n%s\n", SDL_GetError());
            goto cleanup5;
        }
    }
cleanup5:
    SDL_free(window_values);
    SDL_free(tmp_buf);
    SDL_free(window_chars);
cleanup4:
    particle_system_free(&particle_system);
cleanup3:
    SDL_DestroyWindow(window);
cleanup2:
    SDL_Quit();
cleanup1:
    return 0;
}