#pragma once
#include <SDL.h>

#include "random.h"

struct particle_system
{
	float* pos_x;
	float* pos_y;
	float* vel_x;
	float* vel_y;
	float* mass;

	SDL_FPoint* points;

	pcg32_random_t rng;
};

bool particle_system_init(struct particle_system* system);
void particle_system_free(struct particle_system* system);

bool particle_system_update(struct particle_system* system);