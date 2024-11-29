#include <SDL.h>

#define EXIT_SUCCESS 0
#define EXIT_FAILURE -1

int main(int argc, char** argv)
{
    (void) argc;
    (void) argv;

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to initialize SDL!\n%s\n", SDL_GetError());
        return EXIT_FAILURE;
    }

    SDL_Window* window = SDL_CreateWindow("Gravity Simulator", 960, 540, 0);
    if (window == NULL) {
        SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to create SDL Window!\n%s\n", SDL_GetError());
        SDL_Quit();
        return EXIT_FAILURE;
    }

    SDL_Renderer* renderer = SDL_CreateRenderer(window, NULL);
    if (renderer == NULL) {
        SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to create SDL Renderer!\n%s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return EXIT_FAILURE;
    }

    bool running = true;
    while (running) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_EVENT_QUIT) {
                running = false;
            }
        }

        SDL_SetRenderDrawColor(renderer, 51, 77, 204, 255);
        SDL_RenderClear(renderer);

        SDL_RenderPresent(renderer);
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return EXIT_SUCCESS;
}