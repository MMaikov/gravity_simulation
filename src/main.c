#include <SDL.h>

#include "config.h"
#include "particle_system.h"
#include "timer.h"
#include "util.h"

#define EXIT_SUCCESS 0
#define EXIT_FAILURE (-1)

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

    int exitcode = EXIT_SUCCESS;

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to initialize SDL!\n%s\n", SDL_GetError());
        exitcode = EXIT_FAILURE;
        goto cleanup1;
    }

    SDL_Window* window = SDL_CreateWindow(WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT, 0);
    if (window == NULL) {
        SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to create SDL Window!\n%s\n", SDL_GetError());
        exitcode = EXIT_FAILURE;
        goto cleanup2;
    }

    struct particle_system particle_system = { 0 };
    if (!particle_system_init(&particle_system, particle_count)) {
        exitcode = EXIT_FAILURE;
        goto cleanup3;
    }

    SDL_Log("Initialized particle system with %d particles", particle_system.num_particles);
    SDL_Log("Number of threads: %d", particle_system.num_threads);

    SDL_Surface* surface = SDL_GetWindowSurface(window);
    if (surface == NULL) {
        SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to get surface!\n%s\n", SDL_GetError());
        exitcode = EXIT_FAILURE;
        goto cleanup4;
    }

    float* window_values = SDL_calloc(WINDOW_WIDTH * WINDOW_HEIGHT, sizeof(*window_values));
    float* tmp_buf = SDL_calloc(WINDOW_WIDTH * WINDOW_HEIGHT, sizeof(*tmp_buf));
    uint8_t* window_chars = SDL_calloc(WINDOW_WIDTH * WINDOW_HEIGHT, sizeof(*window_chars));

    if (window_values == NULL || tmp_buf == NULL || window_chars == NULL) {
        SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to allocate memory!\n%s\n", SDL_GetError());
        exitcode = EXIT_FAILURE;
        goto cleanup4;
    }

    const SDL_Rect window_rectangle = { .x = 0, .y = 0, .w = WINDOW_WIDTH, .h = WINDOW_HEIGHT };
    const SDL_PixelFormatDetails* pixel_format_details = SDL_GetPixelFormatDetails(surface->format);
    if (pixel_format_details == NULL) {
        SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to get pixel format details!\n%s\n", SDL_GetError());
        exitcode = EXIT_FAILURE;
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

    bool simulate = true;
    int32_t num_updates = 1;
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

        timer_start(&update_timer);
        // TODO: Pass num_updates to the particle system
        for (uint32_t i = 0; i < num_updates && simulate; i++) {
            particle_system_update(&particle_system, dt);
        }
        timer_stop(&update_timer);

        timer_start(&render_timer);

        SDL_FillSurfaceRect(surface, &window_rectangle, SDL_MapRGB(pixel_format_details, NULL, 0, 0, 0));

        for (size_t i = 0; i < particle_system.num_particles; i++) {
            int32_t x = (int32_t)((particle_system.pos_x[i] - view_pos_x)*view_scale);
            int32_t y = (int32_t)((particle_system.pos_y[i] - view_pos_y)*view_scale);
            x = SDL_clamp(x + WINDOW_WIDTH/2, 0, WINDOW_WIDTH-1);
            y = SDL_clamp(y + WINDOW_HEIGHT/2, 0, WINDOW_HEIGHT-1);
            window_values[x + y*WINDOW_WIDTH] += 1;
        }

        blur5x5(window_values, WINDOW_WIDTH, WINDOW_HEIGHT, tmp_buf);

        const float view_brightness = view_scale*brightness;
        for (size_t i = 0; i < WINDOW_WIDTH*WINDOW_HEIGHT-1; i++) {
            float value = window_values[i] * view_brightness;
            window_values[i] = 0.0f;
            value = SDL_min(value, 255.0f);
            set_pixel(surface, (int)i, 0, 0x01010101u * (int)value);
            window_chars[i] = (uint8_t)value;
        }

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

        SDL_UpdateWindowSurface(window);

#if 0
        if (!SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255)) {
            SDL_LogCritical(SDL_LOG_CATEGORY_RENDER, "Failed to set render draw color!\n%s\n", SDL_GetError());
            goto cleanup5;
        }

        if (!SDL_RenderClear(renderer)) {
            SDL_LogCritical(SDL_LOG_CATEGORY_RENDER, "Failed to clear the rendering target!\n%s\n", SDL_GetError());
            goto cleanup5;
        }

        if (!SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255)) {
            SDL_LogCritical(SDL_LOG_CATEGORY_RENDER, "Failed to set render draw color!\n%s\n", SDL_GetError());
            goto cleanup5;
        }

        if (!SDL_RenderPoints(renderer, particle_system.points, (int)particle_system.num_particles)) {
            SDL_LogCritical(SDL_LOG_CATEGORY_RENDER, "Failed to draw points!\n%s\n", SDL_GetError());
            goto cleanup5;
        }

        if (!SDL_RenderPresent(renderer)) {
            SDL_LogCritical(SDL_LOG_CATEGORY_RENDER, "Failed to present the rendering target!\n%s\n", SDL_GetError());
            goto cleanup5;
        }
#endif
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
    return exitcode;
}