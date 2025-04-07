#pragma once

#include "particle_system.h"

void write_to_window_buffer_avx512(float* window_values, struct particle_system* particle_system, float view_pos_x, float view_pos_y, float view_scale);

void write_to_surface_avx512(SDL_Surface* surface, float view_scale, float brightness, float* window_values, uint8_t* window_chars);
