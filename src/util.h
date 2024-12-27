#pragma once

#include <SDL.h>

void set_pixel(SDL_Surface *surface, uint32_t x, uint32_t y, Uint32 pixel);
void blur5x5(float* values, uint32_t width, uint32_t height, float* tmp_buf);