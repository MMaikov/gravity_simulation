#pragma once

#include "main_scalar.h"

#include "util.h"

void write_to_window_buffer_scalar(float* window_values, struct particle_system* particle_system, float view_pos_x, float view_pos_y, float view_scale)
{
    for (size_t i = 0; i < particle_system->num_particles; i++) {
        int32_t x = (int32_t)((particle_system->pos_x[i] - view_pos_x)*view_scale);
        int32_t y = (int32_t)((particle_system->pos_y[i] - view_pos_y)*view_scale);
        x = SDL_clamp(x + WINDOW_WIDTH/2, 0, WINDOW_WIDTH-1);
        y = SDL_clamp(y + WINDOW_HEIGHT/2, 0, WINDOW_HEIGHT-1);
        window_values[x + y*WINDOW_WIDTH] += 1.0f;
    }
}

void write_to_surface_scalar(SDL_Surface* surface, float view_scale, float brightness, float* window_values, uint8_t* window_chars)
{
    const float view_brightness = view_scale*brightness;
    for (size_t i = 0; i < WINDOW_WIDTH*WINDOW_HEIGHT-1; i++) {
        int value = (int)(window_values[i] * view_brightness);
        window_values[i] = 0.0f;
        value = SDL_min(value, 255);
        set_pixel(surface, i, 0, 0x01010101u * value);
        window_chars[i] = (uint8_t)value;
    }
}