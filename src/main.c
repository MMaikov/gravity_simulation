#include <SDL.h>

#include "config.h"
#include "particle_system.h"
#include "timer.h"
#include "util.h"

#include "video_encoder.h"

#include "main_scalar.h"

#if USE_SIMD
#if defined(__AVX512F__) && USE_AVX512
#include "main_avx512.h"
#endif
#if defined(__AVX2__) && USE_AVX
#include "main_avx2.h"
#endif
#endif

void (*write_to_window_buffer)(float* window_values, struct particle_system* particle_system, float view_pos_x, float view_pos_y, float view_scale);
void (*write_to_surface)(SDL_Surface* surface, float view_scale, float brightness, float* window_values, uint8_t* window_chars);

static void init_interface()
{
    write_to_window_buffer = write_to_window_buffer_scalar;
    write_to_surface = write_to_surface_scalar;

#if USE_SIMD

#if defined(__AVX2__) && USE_AVX
    if (SDL_HasAVX2()) {
        write_to_surface = write_to_surface_avx2;
        write_to_window_buffer = write_to_window_buffer_avx2;
    }
#endif

#if defined(__AVX512F__) && USE_AVX512
    if (SDL_HasAVX512F()) {
        write_to_window_buffer = write_to_window_buffer_avx512;
        write_to_surface = write_to_surface_avx512;
    }
#endif

#endif
}

int main(const int argc, char** argv)
{
    uint32_t particle_count = NUM_PARTICLES;

    // TODO: Proper command line argument parsing
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

    init_interface();

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

    uint64_t last_time = SDL_GetTicks();
    uint64_t current_time = 0;

    int32_t num_updates = 1;
    uint32_t img_num = 0;
    float dt = 1e-6f;

    char filename[25] = {0};
    SDL_Time time;
    if (!SDL_GetCurrentTime(&time)) {
        SDL_LogError(SDL_LOG_CATEGORY_ERROR, "Failed to get current time!\n%s\n", SDL_GetError());
        goto cleanup5;
    }

    SDL_DateTime date_time;
    if (!SDL_TimeToDateTime(time, &date_time, true)) {
        SDL_LogError(SDL_LOG_CATEGORY_ERROR, "Failed to get time to datetime!\n%s\n", SDL_GetError());
        goto cleanup5;
    }

    int bytes_written = SDL_snprintf(filename, COUNT_OF(filename), "%02d.%02d.%04d-%02d.%02d.%02d.mkv",
            date_time.day, date_time.month, date_time.year, date_time.hour, date_time.minute, date_time.second);
    if (bytes_written < 23) {
        SDL_Log("%d", bytes_written);
        SDL_LogError(SDL_LOG_CATEGORY_ERROR, "Failed to snprintf or truncated!\n");
        goto cleanup5;
    }

    bool is_video_initialized = false;
    struct video_encoder video_encoder = { 0 };

    SDL_Event e;
    bool running = true;
    while (running) {
        (void) SDL_GetMouseState(&mouse_pos_x, &mouse_pos_y);
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
                    case SDLK_H:
                        timer_reset(&update_timer);
                        timer_reset(&render_timer);
                        break;

                    default: break;
                }
            }
        }

        if (simulate) {
            timer_start(&update_timer);
            for (int32_t i = 0; i < num_updates; i++) {
                particle_system_update(&particle_system, dt);
            }
            timer_stop(&update_timer);
        } else {
            timer_reset(&update_timer);
        }

        timer_start(&render_timer);

        if (!SDL_FillSurfaceRect(surface, &window_rectangle, SDL_MapRGB(pixel_format_details, NULL, 0, 0, 0))) {
            SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to fill surface!\n%s\n", SDL_GetError());
            goto cleanup6;
        }

        write_to_window_buffer(window_values, &particle_system, view_pos_x, view_pos_y, view_scale);

        blur5x5(window_values, WINDOW_WIDTH, WINDOW_HEIGHT, tmp_buf);

        write_to_surface(surface, view_scale, brightness, window_values, window_chars);

        current_time = SDL_GetTicks();
        if (record /*&& (current_time - last_time) >= 55*/) {
            last_time = current_time;
#if VIDEO_OUTPUT
            if (!is_video_initialized) {
                if (0 > video_encoder_init(&video_encoder, filename)) {
                    SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to initialize video encoder!\n");
                    goto cleanup5;
                }

                SDL_Log("Video encoding initialized");
                is_video_initialized = true;
            }
            video_encoder_send_frame(&video_encoder, window_chars);
#else
            // TODO: Multithreaded image saving
            if (!save_image(window_chars, WINDOW_WIDTH, WINDOW_HEIGHT, img_num)) {
                SDL_Log("Failed to save image!");
                record = false;
            }
            img_num++;
#endif
        }

        timer_stop(&render_timer);

        char update_time_buf[10] = {0};
        (void) timer_elapsed_str(&update_timer, COUNT_OF(update_time_buf), update_time_buf);
        char render_time_buf[10] = {0};
        (void) timer_elapsed_str(&render_timer, COUNT_OF(render_time_buf), render_time_buf);
        char title_buf[120] = {0};
        (void) SDL_snprintf(title_buf, COUNT_OF(title_buf), "%s - update: %s render: %s updates: %d dt: %.10e",
            WINDOW_TITLE, update_time_buf, render_time_buf, num_updates, dt);

        if (!SDL_SetWindowTitle(window, title_buf)) {
            SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to set window title!\n%s\n", SDL_GetError());
            goto cleanup6;
        }

        if (!SDL_UpdateWindowSurface(window)) {
            SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to update window surface!\n%s\n", SDL_GetError());
            goto cleanup6;
        }
    }
cleanup6:
    if (is_video_initialized) {
        SDL_Log("Finishing video encoding");
        video_encoder_finish(&video_encoder);
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