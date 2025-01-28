#include "particle_system.h"

#include <SDL.h>

#include "config.h"
#include "random.h"

#if defined(__AVX512F__) && USE_AVX512
#define SIMD_MEMORY_ALIGNMENT 64
#elif defined(__AVX__) && USE_AVX
#define SIMD_MEMORY_ALIGNMENT 32
#elif defined(__SSE__)
#define SIMD_MEMORY_ALIGNMENT 16
#else
#error SIMD not supported
#endif

#include "particle_system_scalar.h"
#include "particle_system_sse.h"
#include "particle_system_avx.h"
#include "particle_system_avx512f.h"

static uint64_t combination(uint64_t n, uint64_t r)
{
	if (r > (n-r)) {
		r = n - r;
	}

	uint64_t result = 1;

	for (size_t i = 0; i < r; ++i) {
		result = result * (n - i) / (i + 1);
	}

	return result;
}

static int particle_thread_func(void* data)
{
	const struct thread_data* thread_data = data;

	void (*attract_particles_batched)(const struct thread_data *thread_data) =
		thread_data->interface->attract_particles_batched;

	if (attract_particles_batched == NULL) {
		SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "attract_particles_batched is null");
		return -1;
	}

	const float rotation = 0.001f * SDL_powf((float)thread_data->system->num_particles, 0.666f);
	SDL_Time time;
	SDL_GetCurrentTime(&time);
	pcg32_random_t rng = PCG32_INITIALIZER;
	pcg32_srandom_r(&rng, time ^ (intptr_t)&SDL_snprintf, (intptr_t)&SDL_CreateCondition ^ 0xDEADBEEF + thread_data->thread_id);

	const float* pos_x = thread_data->system->pos_x;
	const float* pos_y = thread_data->system->pos_y;
	float* vel_x = thread_data->system->buffer.vel_x[thread_data->thread_id];
	float* vel_y = thread_data->system->buffer.vel_y[thread_data->thread_id];

	while (1) {
		if (SDL_GetAtomicInt(thread_data->exit_flag)) {
			break;
		}

		SDL_WaitSemaphore(thread_data->work_start);

		if (SDL_GetAtomicInt(thread_data->exit_flag)) {
			break;
		}

#if PARTICLE_SYSTEM_ADD_RANDOM_FORCE
		for (size_t i = thread_data->particle_range.start; i < thread_data->particle_range.end; ++i) {
			vel_x[i] += pos_y[i]*rotation + ((float)pcg32_random_r(&rng) / (float)SDL_MAX_UINT32 * 2.0f - 1.0f);
			vel_y[i] += -pos_x[i]*rotation + ((float)pcg32_random_r(&rng) / (float)SDL_MAX_UINT32 * 2.0f - 1.0f);
		}
#endif

		attract_particles_batched(thread_data);

		SDL_SignalSemaphore(thread_data->work_done);
	}

	return 0;
}

static bool allocate_memory(struct particle_system* system, const uint32_t num_particles)
{
	system->num_particles = num_particles;

	const size_t PARTICLES_BYTELENGTH = (system->num_particles + AVX512_FLOATS) * sizeof(float);
	system->pos_x  = SDL_aligned_alloc(SIMD_MEMORY_ALIGNMENT, PARTICLES_BYTELENGTH);
	system->pos_y  = SDL_aligned_alloc(SIMD_MEMORY_ALIGNMENT, PARTICLES_BYTELENGTH);
	system->vel_x  = SDL_aligned_alloc(SIMD_MEMORY_ALIGNMENT, PARTICLES_BYTELENGTH);
	system->vel_y  = SDL_aligned_alloc(SIMD_MEMORY_ALIGNMENT, PARTICLES_BYTELENGTH);
	system->mass   = SDL_aligned_alloc(SIMD_MEMORY_ALIGNMENT, PARTICLES_BYTELENGTH);

	if (system->pos_x  == NULL ||
		system->pos_y  == NULL ||
		system->vel_x  == NULL ||
		system->vel_y  == NULL ||
		system->mass   == NULL)
	{
		return false;
	}

#if USE_MULTITHREADING
	system->num_threads = SDL_min(SDL_GetNumLogicalCPUCores(), MAX_THREADS);
	for (size_t i = 0; i < system->num_threads; ++i) {
		float** vel_x = system->buffer.vel_x + i;
		float** vel_y = system->buffer.vel_y + i;

		(*vel_x)  = SDL_aligned_alloc(SIMD_MEMORY_ALIGNMENT, PARTICLES_BYTELENGTH);
		(*vel_y)  = SDL_aligned_alloc(SIMD_MEMORY_ALIGNMENT, PARTICLES_BYTELENGTH);
		if ((*vel_x) == NULL || (*vel_y) == NULL) {
			SDL_aligned_free(system->pos_x);
			SDL_aligned_free(system->pos_y);
			SDL_aligned_free(system->vel_x);
			SDL_aligned_free(system->vel_y);
			SDL_aligned_free(system->mass);
			return false;
		}
	}
#endif

	return true;
}

static void initialize_particles(struct particle_system* system)
{
	const float rotation = 10.0f * SDL_powf((float)system->num_particles, 0.666f);

	for (size_t i = 0; i < system->num_particles+16; ++i) {
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

#if USE_MULTITHREADING
		for (size_t thread = 0; thread < system->num_threads; ++thread) {
			system->buffer.vel_x[thread][i] = 0.0f;
			system->buffer.vel_y[thread][i] = 0.0f;
		}
#endif
	}
}

static bool generate_particle_pairs(struct particle_system* system)
{
#if USE_SIMD

#if defined(__AVX512F__) && USE_AVX512
	const uint32_t SIMD_MULTIPLE = AVX512_FLOATS;
#elif defined(__AVX__) && USE_AVX
	const uint32_t SIMD_MULTIPLE = AVX_FLOATS;
#elif defined(__SSE__)
	const uint32_t SIMD_MULTIPLE = SSE_FLOATS;
#else
#error SIMD not supported
#endif

	size_t it = 0;

	const uint32_t use_particles = SIMD_MULTIPLE*(system->num_particles/SIMD_MULTIPLE);
	system->pairs_simd_length = (use_particles - SIMD_MULTIPLE) * (use_particles / SIMD_MULTIPLE + 1);
	system->pairs_simd = SDL_calloc(system->pairs_simd_length, sizeof(*system->pairs_simd));

	system->pairs_length = SIMD_MULTIPLE*system->num_particles;
	system->pairs      = SDL_calloc(system->pairs_length, sizeof(*system->pairs));

	if (system->pairs_simd == NULL || system->pairs == NULL) {
		SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to allocate memory!\n");
		return false;
	}

	for (size_t hop = SIMD_MULTIPLE; hop < use_particles; ++hop) {
		for (size_t n = 0; n < use_particles/SIMD_MULTIPLE+1; ++n) {
			const size_t i = n*SIMD_MULTIPLE;
			const size_t j = (i + hop) % (use_particles);

			const struct particle_pair pair = {.i = i, .j = j};
			system->pairs_simd[it++] = pair;
		}
	}

	it = 0;

	for (size_t k = 0; k < SIMD_MULTIPLE; ++k) {
		for (size_t n = 0; n < system->num_particles; ++n) {
			const size_t i = n;
			const size_t j = (n + k) % system->num_particles;

			const struct particle_pair pair = {.i = i, .j = j};
			system->pairs[it++] = pair;
		}
	}

#else

	system->pairs_length = combination(system->num_particles, 2);
	system->pairs      = SDL_calloc(system->pairs_length, sizeof(*system->pairs));

	system->pairs_simd = NULL;
	system->pairs_simd_length = 0;

	if (system->pairs == NULL) {
		SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to allocate memory!\n");
		return false;
	}

	size_t it = 0;
	for (size_t i = 0; i < system->num_particles; ++i) {
		for (size_t j = i+1; j < system->num_particles; ++j) {
			const struct particle_pair pair = {.i = i, .j = j};
			system->pairs[it++] = pair;
		}
	}
#endif

	return true;
}

static bool initialize_threads(struct particle_system* system)
{
	system->work_start = SDL_CreateSemaphore(0);
	system->work_done = SDL_CreateSemaphore(0);

	if (system->work_start == NULL || system->work_done == NULL) {
		SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to create SDL semaphores!\n%s", SDL_GetError());
		return false;
	}

	// Allow threads to start
	(void) SDL_SetAtomicInt(&system->exit_flag, 0);

	size_t st = 0;
	size_t nd = system->pairs_length/system->num_threads;
	size_t st2 = 0;
	size_t nd2 = system->pairs_simd_length/system->num_threads;
	size_t st3 = 0;
	size_t nd3 = (system->num_particles+system->num_threads) / system->num_threads;
	for (size_t i = 0; i < system->num_threads; ++i) {
		const struct range simd_range = {.start = st2, .end = nd2 };
		const struct range scalar_range = {.start = st, .end = nd };
		const struct range particle_range = {.start = st3, .end = nd3 };
		const struct thread_data data = {.system = system, .dt = 1e-6f, .thread_id = i, .simd_range = simd_range,
			.scalar_range = scalar_range, .particle_range = particle_range, .work_start = system->work_start,
			.work_done = system->work_done, .exit_flag = &system->exit_flag,
			.interface = &system->interface};
		st += system->pairs_length/system->num_threads;
		nd += system->pairs_length/system->num_threads;
		st2 += system->pairs_simd_length/system->num_threads;
		nd2 += system->pairs_simd_length/system->num_threads;
		st3 += (system->num_particles+system->num_threads) / system->num_threads;
		nd3 += (system->num_particles+system->num_threads) / system->num_threads;

		system->thread_data[i] = data;
		system->threads[i] = SDL_CreateThread(particle_thread_func, "ParticleWorker", system->thread_data + i);
		if (system->threads[i] == NULL) {
			SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to create a SDL thread!\n%s", SDL_GetError());
			return false;
		}
	}

	return true;
}

static void set_interface(struct particle_system* system) {
#if USE_SIMD
#if defined(__AVX512F__) && USE_AVX512
	const struct interface interface = {
		.attract_particles = &attract_particles_avx512,
		.attract_particles_batched = &attract_particles_avx512_batched,
		.move_particles = &move_particles_avx512,
		.calculate_average = &calculate_average_avx512,
		.calculate_standard_distribution = &calculate_standard_distribution_avx512,
		.expand_universe = &expand_universe_avx512,
		.copy_multithreaded_velocities = &copy_multithreaded_velocities_avx512,
	};
#elif defined(__AVX__) && USE_AVX
	const struct interface interface = {
		.attract_particles = &attract_particles_avx,
		.attract_particles_batched = &attract_particles_avx_batched,
		.move_particles = &move_particles_avx,
		.calculate_average = &calculate_average_avx,
		.calculate_standard_distribution = &calculate_standard_distribution_avx,
		.expand_universe = &expand_universe_avx,
		.copy_multithreaded_velocities = &copy_multithreaded_velocities_avx,
	};
#elif defined(__SSE__)
	const struct interface interface = {
		.attract_particles = &attract_particles_sse,
		.attract_particles_batched = &attract_particles_sse_batched,
		.move_particles = &move_particles_sse,
		.calculate_average = &calculate_average_sse,
		.calculate_standard_distribution = &calculate_standard_distribution_sse,
		.expand_universe = &expand_universe_sse,
		.copy_multithreaded_velocities = &copy_multithreaded_velocities_sse,
	};
#else
#error SIMD not supported
#endif
#else
	const struct interface interface = {
		.attract_particles = &attract_particles_scalar,
		.attract_particles_batched = &attract_particles_scalar_batched,
		.move_particles = &move_particles_scalar,
		.calculate_average = &calculate_average_scalar,
		.calculate_standard_distribution = &calculate_standard_distribution_scalar,
		.expand_universe = &expand_universe_scalar,
		.copy_multithreaded_velocities = &copy_multithreaded_velocities_scalar,
	};
#endif
	system->interface = interface;
}

bool particle_system_init(struct particle_system* system, const uint32_t num_particles)
{
	set_interface(system);

	if (!allocate_memory(system, num_particles)) {
		SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to allocate memory!\n");
		return false;
	}

	pcg32_srandom(&system->rng);

	initialize_particles(system);

#if USE_MULTITHREADING
	if (!generate_particle_pairs(system)) {
		return false;
	}

	if (!initialize_threads(system)) {
		return false;
	}
#endif

	return true;
}

void particle_system_reset(struct particle_system* system) {
	initialize_particles(system);
}

void particle_system_free(struct particle_system* system)
{
#if USE_MULTITHREADING
	(void) SDL_SetAtomicInt(&system->exit_flag, 1);

	// Wake up any threads waiting on work_start
	for (uint32_t i = 0; i < system->num_threads; i++) {
		SDL_SignalSemaphore(system->work_start);
	}

	for (uint32_t i = 0; i < system->num_threads; ++i) {
		SDL_WaitThread(system->threads[i], NULL);
	}

	SDL_DestroySemaphore(system->work_start);
	SDL_DestroySemaphore(system->work_done);

	for (uint32_t i = 0; i < system->num_threads; ++i) {
		SDL_aligned_free(system->buffer.vel_x[i]);
		SDL_aligned_free(system->buffer.vel_y[i]);
	}

	SDL_free(system->pairs);
	SDL_free(system->pairs_simd);
#endif

	SDL_aligned_free(system->pos_x);
	SDL_aligned_free(system->pos_y);
	SDL_aligned_free(system->vel_x);
	SDL_aligned_free(system->vel_y);
	SDL_aligned_free(system->mass );
}

void particle_system_update(struct particle_system* system, float dt)
{
#if USE_MULTITHREADING
	for (size_t thread = 0; thread < system->num_threads; ++thread) {
		system->thread_data[thread].dt = dt;
	}
#endif

	const float amount = 0.7f / system->interface.calculate_standard_distribution(system);
	system->interface.expand_universe(system, amount);

#if USE_MULTITHREADING

	// Signal all threads to start work
	for (uint32_t i = 0; i < system->num_threads; i++) {
		SDL_SignalSemaphore(system->work_start);
	}

	// Wait for all threads to finish
	for (uint32_t i = 0; i < system->num_threads; i++) {
		SDL_WaitSemaphore(system->work_done);
	}

#else
	system->interface.attract_particles(system, dt);
#endif

#if USE_MULTITHREADING
	system->interface.copy_multithreaded_velocities(system);
#endif

	system->interface.move_particles(system, dt);
}