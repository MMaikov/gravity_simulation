#pragma once

#include <SDL.h>

void set_pixel(SDL_Surface *surface, uint32_t x, uint32_t y, Uint32 pixel);
void blur5x5(float* values, uint32_t width, uint32_t height, float* tmp_buf);
bool save_image(uint8_t* pixels, uint32_t width, uint32_t height, uint32_t name_index);