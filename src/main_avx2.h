#pragma once

#include <SDL.h>

void write_to_surface_avx2(SDL_Surface* surface, float view_scale, float brightness, float* window_values, uint8_t* window_chars);
