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

struct thread_data
{
	struct particle_system* system;
	float dt;
	uint32_t num_updates;
	size_t thread_id;
	size_t simd_start;
	size_t simd_end;
	size_t start;
	size_t end;

	SDL_Semaphore* work_start;
	SDL_Semaphore* work_done;
	SDL_AtomicInt* exit_flag;
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

	int num_threads;
	uint32_t num_particles;
};

bool particle_system_init(struct particle_system* system, uint32_t num_particles);
void particle_system_free(struct particle_system* system);

void particle_system_update(struct particle_system* system, float dt, uint32_t num_updates);