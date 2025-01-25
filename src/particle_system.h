#pragma once
#include <SDL.h>

#include "random.h"
#include "config.h"

struct particles_buffer
{
	float* vel_x[MAX_THREADS];
	float* vel_y[MAX_THREADS];
};

struct particle_pair
{
	size_t i;
	size_t j;
};

struct particle_system;

struct range {
	size_t start;
	size_t end;
};

struct thread_data
{
	struct particle_system* system;
	struct interface* interface;
	float dt;
	size_t thread_id;
	struct range simd_range;
	struct range scalar_range;

	SDL_Semaphore* work_start;
	SDL_Semaphore* work_done;
	SDL_AtomicInt* exit_flag;
};

struct interface
{
	void (*attract_particles)(struct particle_system *system, const float dt);
	void (*attract_particles_batched)(const struct thread_data* thread_data);
	void (*move_particles)(struct particle_system *system, const float dt);
	void (*calculate_average)(struct particle_system* system, float* out_avg_x, float* out_avg_y);
	float (*calculate_standard_distribution)(struct particle_system* system);
	void (*expand_universe)(struct particle_system* system, const float amount);
	void (*copy_multithreaded_velocities)(struct particle_system* system);

	bool (*generate_particle_pairs)(struct particle_system* system);
};

struct particle_system
{
	float* pos_x;
	float* pos_y;
	float* vel_x;
	float* vel_y;
	float* mass;

	pcg32_random_t rng;

	struct particles_buffer buffer;

	size_t pairs_length;
	struct particle_pair* pairs;

	size_t pairs_simd_length;
	struct particle_pair* pairs_simd;

	SDL_Thread* threads[MAX_THREADS];
	struct thread_data thread_data[MAX_THREADS];
	SDL_Semaphore* work_start;
	SDL_Semaphore* work_done;
	SDL_AtomicInt exit_flag;

	uint32_t num_threads;
	uint32_t num_particles;

	struct interface interface;
};

bool particle_system_init(struct particle_system* system, uint32_t num_particles);
void particle_system_free(struct particle_system* system);

void particle_system_reset(struct particle_system* system);

void particle_system_update(struct particle_system* system, float dt);