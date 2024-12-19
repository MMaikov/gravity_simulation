#include "particle_system.h"

#include <SDL.h>

#include "config.h"

#define SIMD_MEMORY_ALIGNMENT 32

static float calculate_standard_distribution(struct particle_system* system);

static void copy_pos_to_points(struct particle_system* system)
{
	for (size_t i = 0; i < MAX_PARTICLES; ++i) {
		system->points[i].x = (system->pos_x[i] + 1.6f) * (WINDOW_HEIGHT / 2.0f);
		system->points[i].y = (system->pos_y[i] + 1.0f) * (WINDOW_HEIGHT / 2.0f);
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

	pcg32_srandom(&system->rng);
	const float rotation = 0.1f * SDL_powf(MAX_PARTICLES, 0.666f);

	for (size_t i = 0; i < MAX_PARTICLES; ++i) {
		const float pos_x = (float)pcg32_random_r(&system->rng) / (float)SDL_MAX_UINT32;
		const float pos_y = (float)pcg32_random_r(&system->rng) / (float)SDL_MAX_UINT32;
		float vel_x = pos_y*rotation;
		float vel_y = -pos_x*rotation;
		const float mass = (float)pcg32_random_r(&system->rng) / (float)SDL_MAX_UINT32 * 10;

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

void particle_system_free(struct particle_system* system)
{
	SDL_aligned_free(system->pos_x);
	SDL_aligned_free(system->pos_y);
	SDL_aligned_free(system->vel_x);
	SDL_aligned_free(system->vel_y);
	SDL_aligned_free(system->mass );
}

static void move_particles(struct particle_system *system, const float dt) {
	for (size_t i = 0; i < MAX_PARTICLES; ++i) {
		system->pos_x[i] += dt * system->vel_x[i];
		system->pos_y[i] += dt * system->vel_y[i];
	}
}

static void attract_particles(struct particle_system *system, const float dt) {
	const float min_dist = 4*dt;
	for (size_t i = 0; i < MAX_PARTICLES; ++i) {
		for (size_t j = i+1; j < MAX_PARTICLES; ++j) {
			const float pos_x_diff = system->pos_x[i] - system->pos_x[j];
			const float pos_y_diff = system->pos_y[i] - system->pos_y[j];
			const float distance_squared = pos_x_diff * pos_x_diff + pos_y_diff * pos_y_diff;
			const float distance = SDL_max(min_dist, SDL_sqrtf(distance_squared));
			const float force = (system->mass[i] * system->mass[j]) / (distance * distance * distance);
			const float force_x = force * pos_x_diff;
			const float force_y = force * pos_y_diff;

			system->vel_x[i] += dt * force_x / system->mass[i];
			system->vel_y[i] += dt * force_y / system->mass[i];

			system->vel_x[j] -= dt * force_x / system->mass[j];
			system->vel_y[j] -= dt * force_y / system->mass[j];
		}
	}
}

static float calculate_standard_distribution(struct particle_system* system)
{
	float x_avg = 0.0f;
	float y_avg = 0.0f;
	for (size_t i = 0; i < MAX_PARTICLES; ++i) {
		x_avg += system->pos_x[i];
		y_avg += system->pos_y[i];
	}
	x_avg /= (float)MAX_PARTICLES;
	y_avg /= (float)MAX_PARTICLES;

	float x_dist = 0.0f;
	float y_dist = 0.0f;
	for (size_t i = 0; i < MAX_PARTICLES; ++i) {
		x_dist += (system->pos_x[i] - x_avg)*(system->pos_x[i] - x_avg);
		y_dist += (system->pos_y[i] - y_avg)*(system->pos_y[i] - y_avg);
	}
	x_dist /= (float)MAX_PARTICLES;
	y_dist /= (float)MAX_PARTICLES;
	return SDL_sqrtf(x_dist + y_dist);
}

static void expand_universe(struct particle_system* system, const float amount)
{
	float avg_x = 0.0f;
	float avg_y = 0.0f;
	for (size_t i = 0; i < MAX_PARTICLES; ++i) {
		avg_x += system->pos_x[i];
		avg_y += system->pos_y[i];
	}
	avg_x /= (float)MAX_PARTICLES;
	avg_y /= (float)MAX_PARTICLES;

	for (size_t i = 0; i < MAX_PARTICLES; ++i) {
		system->pos_x[i] = amount * (system->pos_x[i] - avg_x);
		system->pos_y[i] = amount * (system->pos_y[i] - avg_y);
	}
}

bool particle_system_update(struct particle_system* system)
{
	const float dt = 1e-5f;

	const float amount = 0.7f / calculate_standard_distribution(system);
	expand_universe(system, amount);

	const float rotation = 0.001f * SDL_powf(MAX_PARTICLES, 0.666f);
	for (size_t i = 0; i < MAX_PARTICLES; ++i) {
		system->vel_x[i] += system->pos_y[i]*rotation + ((float)pcg32_random_r(&system->rng) / (float)SDL_MAX_UINT32 * 2.0f - 1.0f);
		system->vel_y[i] += -system->pos_x[i]*rotation + ((float)pcg32_random_r(&system->rng) / (float)SDL_MAX_UINT32 * 2.0f - 1.0f);
	}

	attract_particles(system, dt);

	move_particles(system, dt);

	copy_pos_to_points(system);

	return true;
}