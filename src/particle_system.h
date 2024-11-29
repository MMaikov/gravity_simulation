#pragma once
#include <SDL.h>

struct particle_system
{
	float* pos_x;
	float* pos_y;
	float* vel_x;
	float* vel_y;
	float* mass;

	SDL_FPoint* points;
};

bool particle_system_init(struct particle_system* system);
bool particle_system_free(struct particle_system* system);

bool particle_system_update(struct particle_system* system);