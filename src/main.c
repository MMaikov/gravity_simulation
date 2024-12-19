#include <SDL.h>

#include "config.h"
#include "particle_system.h"

#define EXIT_SUCCESS 0
#define EXIT_FAILURE (-1)

int main(const int argc, char** argv)
{
    (void) argc;
    (void) argv;

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

    SDL_Renderer* renderer = SDL_CreateRenderer(window, NULL);
    if (renderer == NULL) {
        SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to create SDL Renderer!\n%s\n", SDL_GetError());
        exitcode = EXIT_FAILURE;
        goto cleanup3;
    }

    struct particle_system particle_system = { 0 };
    if (!particle_system_init(&particle_system)) {
        exitcode = EXIT_FAILURE;
        goto cleanup4;
    }

    SDL_Log("Initialized particle system with %d particles", MAX_PARTICLES);

    float sum = 0.0f;
    uint32_t count = 0;

    bool running = true;
    while (running) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_EVENT_QUIT) {
                running = false;
            }
        }

        const Uint64 start = SDL_GetPerformanceCounter();
        particle_system_update(&particle_system);
        const Uint64 end = SDL_GetPerformanceCounter();
        const float elapsed = (float)(end - start) / (float)SDL_GetPerformanceFrequency() * 1000.0f * 1000.0f;
        sum += elapsed;
        count += 1;

        const float average = sum / (float)count;
        
        char buf[40] = {0};
        SDL_snprintf(buf, 40, "%s - %.1f us", WINDOW_TITLE, average);

        if (!SDL_SetWindowTitle(window, buf)) {
            SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to set window title!\n%s\n", SDL_GetError());
            goto cleanup5;
        }

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

        if (!SDL_RenderPoints(renderer, particle_system.points, MAX_PARTICLES)) {
            SDL_LogCritical(SDL_LOG_CATEGORY_RENDER, "Failed to draw points!\n%s\n", SDL_GetError());
            goto cleanup5;
        }

        if (!SDL_RenderPresent(renderer)) {
            SDL_LogCritical(SDL_LOG_CATEGORY_RENDER, "Failed to present the rendering target!\n%s\n", SDL_GetError());
            goto cleanup5;
        }
    }

cleanup5:
    particle_system_free(&particle_system);
cleanup4:
    SDL_DestroyRenderer(renderer);
cleanup3:
    SDL_DestroyWindow(window);
cleanup2:
    SDL_Quit();
cleanup1:
    return exitcode;
}