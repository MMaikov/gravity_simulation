#include <SDL.h>

#define SDL_MAIN_USE_CALLBACKS
#include <SDL3/SDL_main.h>

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

struct vec2 {
    float x;
    float y;
};

typedef enum {
    APP_FLAG_NONE             = 0,
    APP_FLAG_VIDEO_INITIALIZED = (1 << 0),
    APP_FLAG_SIMULATE          = (1 << 1),
    APP_FLAG_RECORD            = (1 << 2)
} AppFlags;

struct simulation_state {
    SDL_Window* window;
    SDL_Surface* surface;

    const SDL_PixelFormatDetails* pixel_format_details;

    struct particle_system particle_system;
    struct video_encoder video_encoder;
    struct timer update_timer;
    struct timer render_timer;

    float* window_values;
    float* tmp_buf;
    uint8_t* window_chars;

    uint8_t flags;

    struct vec2 view_pos;
    float view_scale;
    float brightness;

    struct vec2 mouse_pos;

    uint64_t last_time;
    uint64_t current_time;

    uint32_t num_updates;
    uint32_t img_num;
    float dt;

    char filename[25];
};

// ReSharper disable once CppParameterMayBeConst
SDL_AppResult SDL_AppInit(void **appstate, int argc, char **argv) {

    struct simulation_state* state = SDL_calloc(1, sizeof(*state));
    if (state == NULL) {
        return SDL_APP_FAILURE;
    }
    *appstate = state;

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
        return SDL_APP_FAILURE;
    }

    init_interface();

    state->window = SDL_CreateWindow(WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT, 0);
    if (state->window == NULL) {
        SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to create SDL Window!\n%s\n", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    if (!particle_system_init(&state->particle_system, particle_count)) {
        SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to initialize particle_system!\n%s\n", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    SDL_Log("Initialized particle system with %d particles", state->particle_system.num_particles);
    SDL_Log("Number of threads: %d", state->particle_system.num_threads);

    state->surface = SDL_GetWindowSurface(state->window);
    if (state->surface == NULL) {
        SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to get surface!\n%s\n", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    state->window_values = SDL_calloc(WINDOW_WIDTH * WINDOW_HEIGHT, sizeof(*state->window_values));
    state->tmp_buf = SDL_calloc(WINDOW_WIDTH * WINDOW_HEIGHT, sizeof(*state->tmp_buf));
    state->window_chars = SDL_calloc(WINDOW_WIDTH * WINDOW_HEIGHT, sizeof(*state->window_chars));

    if (state->window_values == NULL || state->tmp_buf == NULL || state->window_chars == NULL) {
        SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to allocate memory!\n%s\n", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    state->pixel_format_details = SDL_GetPixelFormatDetails(state->surface->format);
    if (state->pixel_format_details == NULL) {
        SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to get pixel format details!\n%s\n", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    state->view_pos.x = 0.0f;
    state->view_pos.y = 0.0f;
    state->view_scale = WINDOW_HEIGHT / 2.0f;
    state->brightness = 10000.0f / (float)state->particle_system.num_particles;

    SDL_GetMouseState(&state->mouse_pos.x, &state->mouse_pos.y);

    state->last_time = SDL_GetTicks();
    state->current_time = 0;

    state->num_updates = 1;
    state->img_num = 0;
    state->dt = 1e-6f;

    SDL_Time time;
    if (!SDL_GetCurrentTime(&time)) {
        SDL_LogError(SDL_LOG_CATEGORY_ERROR, "Failed to get current time!\n%s\n", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    SDL_DateTime date_time;
    if (!SDL_TimeToDateTime(time, &date_time, true)) {
        SDL_LogError(SDL_LOG_CATEGORY_ERROR, "Failed to get time to datetime!\n%s\n", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    const int bytes_written = SDL_snprintf(state->filename, COUNT_OF(state->filename),
        "%02d.%02d.%04d-%02d.%02d.%02d.mkv", date_time.day, date_time.month, date_time.year,
            date_time.hour, date_time.minute, date_time.second);
    if (bytes_written < 23) {
        SDL_LogError(SDL_LOG_CATEGORY_ERROR, "Failed to snprintf or truncated!\n"
                                             "Wrote %d bytes, but expected 23 bytes to be written (excl. \\0)\n",
                                             bytes_written);
        return SDL_APP_FAILURE;
    }

    return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppIterate(void *appstate) {

    struct simulation_state* state = appstate;

    SDL_GetMouseState(&state->mouse_pos.x, &state->mouse_pos.y);

    if (state->flags & APP_FLAG_SIMULATE) {
        timer_start(&state->update_timer);
        for (uint32_t i = 0; i < state->num_updates; i++) {
            particle_system_update(&state->particle_system, state->dt);
        }
        timer_stop(&state->update_timer);
    } else {
        timer_reset(&state->update_timer);
    }

    timer_start(&state->render_timer);

    if (!SDL_FillSurfaceRect(state->surface, NULL,
        SDL_MapRGB(state->pixel_format_details, NULL, 0, 0, 0))) {
        SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to fill surface!\n%s\n", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    write_to_window_buffer(state->window_values, &state->particle_system,
        state->view_pos.x, state->view_pos.y, state->view_scale);

    blur5x5(state->window_values, WINDOW_WIDTH, WINDOW_HEIGHT, state->tmp_buf);

    write_to_surface(state->surface, state->view_scale, state->brightness, state->window_values, state->window_chars);

    state->current_time = SDL_GetTicks();
    if (state->flags & APP_FLAG_RECORD) {
        state->last_time = state->current_time;
#if VIDEO_OUTPUT
        if (!(state->flags & APP_FLAG_VIDEO_INITIALIZED)) {
            if (video_encoder_init(&state->video_encoder, state->filename) < 0) {
                SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to initialize video encoder!\n");
                return SDL_APP_FAILURE;
            }

            SDL_Log("Video encoding initialized");
            state->flags |= APP_FLAG_VIDEO_INITIALIZED;
        }
        video_encoder_send_frame(&state->video_encoder, state->window_chars);
#else
        // TODO: Multithreaded image saving
        if (!save_image(state->window_chars, WINDOW_WIDTH, WINDOW_HEIGHT, state->img_num)) {
            SDL_Log("Failed to save image!");
            state->flags &= ~APP_FLAG_RECORD;
        }
        state->img_num++;
#endif
    }

    timer_stop(&state->render_timer);

    char update_time_buf[10] = {0};
    (void) timer_elapsed_str(&state->update_timer, COUNT_OF(update_time_buf), update_time_buf);
    char render_time_buf[10] = {0};
    (void) timer_elapsed_str(&state->render_timer, COUNT_OF(render_time_buf), render_time_buf);
    char title_buf[120] = {0};
    (void) SDL_snprintf(title_buf, COUNT_OF(title_buf), "%s - update: %s render: %s updates: %d dt: %.10e",
        WINDOW_TITLE, update_time_buf, render_time_buf, state->num_updates, state->dt);

    if (!SDL_SetWindowTitle(state->window, title_buf)) {
        SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to set window title!\n%s\n", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    if (!SDL_UpdateWindowSurface(state->window)) {
        SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to update window surface!\n%s\n", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    return SDL_APP_CONTINUE;
}

static bool handle_view_controls(struct simulation_state* state, const SDL_KeyboardEvent* event) {
    const float factor = state->view_scale * 0.01f;
    switch (event->key) {
        case SDLK_LEFT:
            state->view_pos.x -= WINDOW_HEIGHT / factor;
            return true;
        case SDLK_RIGHT:
            state->view_pos.x += WINDOW_WIDTH / factor;
            return true;
        case SDLK_UP:
            state->view_pos.y -= WINDOW_HEIGHT / factor;
            return true;
        case SDLK_DOWN:
            state->view_pos.y += WINDOW_HEIGHT / factor;
            return true;
        case SDLK_O:
            state->view_scale *= 0.98f;
            return true;
        case SDLK_I:
            state->view_scale *= 1.02f;
            return true;
        case SDLK_E:
            state->brightness *= 1.05f;
            return true;
        case SDLK_W:
            state->brightness *= 0.95f;
            return true;
        default: return false;
    }
}

static bool handle_simulation_controls(struct simulation_state* state, const SDL_KeyboardEvent* event) {
    switch (event->key) {
        case SDLK_SPACE:
            state->flags ^= APP_FLAG_SIMULATE;
            return true;
        case SDLK_R:
            particle_system_reset(&state->particle_system);
            timer_reset(&state->update_timer);
            state->flags &= ~APP_FLAG_SIMULATE;
            return true;
        case SDLK_S:
            state->flags |= APP_FLAG_SIMULATE | APP_FLAG_RECORD;
            return true;
        case SDLK_F:
            state->num_updates += 1;
            return true;
        case SDLK_G:
            state->num_updates -= 1;
            state->num_updates = SDL_max(state->num_updates, 0);
            return true;
        case SDLK_V:
            state->dt *= 1.1f;
            return true;
        case SDLK_B:
            state->dt *= 1.0f / 1.1f;
            return true;
        case SDLK_H:
            timer_reset(&state->update_timer);
            timer_reset(&state->render_timer);
            return true;
        default: return false;
    }
}


static SDL_AppResult handle_key_down(struct simulation_state* state, const SDL_KeyboardEvent* event) {

    if (event->key == SDLK_ESCAPE) {
        return SDL_APP_SUCCESS;
    }

    if (handle_view_controls(state, event)) {
        return SDL_APP_CONTINUE;
    }

    if (handle_simulation_controls(state, event)) {
        return SDL_APP_CONTINUE;
    }

    return SDL_APP_CONTINUE;
}

// ReSharper disable once CppParameterMayBeConstPtrOrRef
SDL_AppResult SDL_AppEvent(void *appstate, SDL_Event *event) {
    struct simulation_state* state = appstate;

    if (event->type == SDL_EVENT_QUIT) {
        return SDL_APP_SUCCESS;
    }

    if (event->type == SDL_EVENT_MOUSE_WHEEL) {
        const float zoom_factor = (event->wheel.y < 0) ? 0.95f : 1.05f;
        const float change_x = state->mouse_pos.x - WINDOW_WIDTH / 2.0f;
        const float change_y = state->mouse_pos.y - WINDOW_HEIGHT / 2.0f;

        state->view_pos.x += change_x / state->view_scale;
        state->view_pos.y += change_y / state->view_scale;

        state->view_scale *= zoom_factor;

        state->view_pos.x = state->view_pos.x - change_x / state->view_scale;
        state->view_pos.y = state->view_pos.y - change_y / state->view_scale;
    } else if (event->type == SDL_EVENT_KEY_DOWN) {
        return handle_key_down(state, &event->key);
    }

    return SDL_APP_CONTINUE;
}

void SDL_AppQuit(void *appstate, SDL_AppResult result) {

    struct simulation_state* state = appstate;

    if (state->flags & APP_FLAG_VIDEO_INITIALIZED) {
        SDL_Log("Finishing video encoding");
        video_encoder_finish(&state->video_encoder);
    }

    SDL_free(state->window_values);
    SDL_free(state->tmp_buf);
    SDL_free(state->window_chars);

    particle_system_free(&state->particle_system);

    SDL_DestroyWindow(state->window);

    SDL_free(state);

    SDL_Quit();
}