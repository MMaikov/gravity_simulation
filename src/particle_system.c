#include "particle_system.h"

#include <SDL.h>

#include "config.h"
#include "random.h"

#define SIMD_MEMORY_ALIGNMENT 32

static void copy_pos_to_points(struct particle_system* system)
{
	for (size_t i = 0; i < MAX_PARTICLES; ++i) {
		system->points[i].x = system->pos_x[i];
		system->points[i].y = system->pos_y[i];
	}
}

bool particle_system_init(struct particle_system* system)
{
	const size_t PARTICLES_LENGTH = MAX_PARTICLES * sizeof(float);

	system->pos_x  = SDL_aligned_alloc(SIMD_MEMORY_ALIGNMENT, PARTICLES_LENGTH);
	system->pos_y  = SDL_aligned_alloc(SIMD_MEMORY_ALIGNMENT, PARTICLES_LENGTH);
	system->vel_x  = SDL_aligned_alloc(SIMD_MEMORY_ALIGNMENT, PARTICLES_LENGTH);
	system->vel_y  = SDL_aligned_alloc(SIMD_MEMORY_ALIGNMENT, PARTICLES_LENGTH);
	system->mass   = SDL_aligned_alloc(SIMD_MEMORY_ALIGNMENT, PARTICLES_LENGTH);
	system->points = SDL_calloc(MAX_PARTICLES, sizeof(*system->points));

	if (system->pos_x  == NULL ||
		system->pos_y  == NULL ||
		system->vel_x  == NULL ||
		system->vel_y  == NULL ||
		system->mass   == NULL ||
		system->points == NULL)
	{
		SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to allocate memory!\n");
		return false;
	}
	
	pcg32_random_t rng = PCG32_INITIALIZER;

	for (size_t i = 0; i < MAX_PARTICLES; ++i) {
		float pos_x = (float)pcg32_random_r(&rng) / (float)SDL_MAX_UINT32 * WINDOW_WIDTH;
		float pos_y = (float)pcg32_random_r(&rng) / (float)SDL_MAX_UINT32 * WINDOW_HEIGHT;
		float vel_x = ((WINDOW_WIDTH / 2) - pos_x)/SDL_abs((WINDOW_WIDTH / 2) - pos_x);
		float vel_y = ((WINDOW_HEIGHT / 2) - pos_y)/SDL_abs((WINDOW_HEIGHT / 2) - pos_y);
		float mass = (float)pcg32_random_r(&rng) / (float)SDL_MAX_UINT32;

		if (SDL_isinff(vel_x)) vel_x = 0.0f;
		if (SDL_isinff(vel_y)) vel_y = 0.0f;

		system->pos_x[i] = pos_x;
		system->pos_y[i] = pos_y;
		system->vel_x[i] = vel_x;
		system->vel_y[i] = vel_y;
		system->mass[i] = mass;
	}

	copy_pos_to_points(system);

	return true;
}

bool particle_system_free(struct particle_system* system)
{
	SDL_aligned_free(system->pos_x);
	SDL_aligned_free(system->pos_y);
	SDL_aligned_free(system->vel_x);
	SDL_aligned_free(system->vel_y);
	SDL_aligned_free(system->mass );
}

bool particle_system_update(struct particle_system* system)
{
	const float dt = 0.1f;

	for (size_t i = 0; i < MAX_PARTICLES; ++i) {
		for (size_t j = i+1; j < MAX_PARTICLES; ++j) {
			float pos_x_diff = system->pos_x[i] - system->pos_x[j];
			float pos_y_diff = system->pos_y[i] - system->pos_y[j];
			float distance_squared = pos_x_diff * pos_x_diff + pos_y_diff * pos_y_diff;
			float distance = SDL_max(dt, SDL_sqrtf(distance_squared));
			float force = (system->mass[i] * system->mass[j]) / (distance * distance * distance);
			float force_x = force * pos_x_diff;
			float force_y = force * pos_y_diff;

			system->vel_x[i] -= dt * force_x / system->mass[i];
			system->vel_y[i] -= dt * force_y / system->mass[i];

			system->vel_x[j] += dt * force_x / system->mass[j];
			system->vel_y[j] += dt * force_y / system->mass[j];
		}
	}

	for (size_t i = 0; i < MAX_PARTICLES; ++i) {
		system->pos_x[i] += dt * system->vel_x[i];
		system->pos_y[i] += dt * system->vel_y[i];
	}

	copy_pos_to_points(system);

	return true;
}