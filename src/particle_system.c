#include "particle_system.h"

#include <SDL.h>

#include "config.h"

#if defined(__AVX512F__) && USE_AVX512
#define SIMD_MEMORY_ALIGNMENT 64
#elif defined(__AVX__) && USE_AVX
#define SIMD_MEMORY_ALIGNMENT 32
#elif defined(__SSE__)
#define SIMD_MEMORY_ALIGNMENT 16
#else
#error SIMD not supported
#endif

#include <immintrin.h>
#include "simd_util.h"

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

static void attract_particles_batched(const struct thread_data* thread_data);
static void attract_particles_sse_batched(const struct thread_data* thread_data);
static void attract_particles_avx_batched(const struct thread_data* thread_data);
static void attract_particles_avx512_batched(const struct thread_data* thread_data);

static int thread_func(void* data)
{
	const struct thread_data* thread_data = (struct thread_data*)data;

	while (1) {
		if (SDL_GetAtomicInt(thread_data->exit_flag)) {
			break;
		}

		SDL_WaitSemaphore(thread_data->work_start);

		if (SDL_GetAtomicInt(thread_data->exit_flag)) {
			break;
		}

		for (uint32_t i = 0; i < thread_data->num_updates; ++i) {
#if USE_SIMD
#if defined(__AVX512F__) && USE_AVX512
			attract_particles_avx512_batched(thread_data);
#elif defined(__AVX__) && USE_AVX
			attract_particles_avx_batched(thread_data);
#elif defined(__SSE__)
			attract_particles_sse_batched(thread_data);
#else
#error SIMD not supported
#endif
#else
			attract_particles_batched(thread_data);
#endif
		}

		SDL_SignalSemaphore(thread_data->work_done);
	}

	return 0;
}

static bool allocate_memory(struct particle_system* system, const uint32_t num_particles)
{
	system->num_particles = num_particles;

	const size_t PARTICLES_LENGTH = (system->num_particles + AVX512_FLOATS) * sizeof(float);
	system->pos_x  = SDL_aligned_alloc(SIMD_MEMORY_ALIGNMENT, PARTICLES_LENGTH);
	system->pos_y  = SDL_aligned_alloc(SIMD_MEMORY_ALIGNMENT, PARTICLES_LENGTH);
	system->vel_x  = SDL_aligned_alloc(SIMD_MEMORY_ALIGNMENT, PARTICLES_LENGTH);
	system->vel_y  = SDL_aligned_alloc(SIMD_MEMORY_ALIGNMENT, PARTICLES_LENGTH);
	system->mass   = SDL_aligned_alloc(SIMD_MEMORY_ALIGNMENT, PARTICLES_LENGTH);

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

		(*vel_x)  = SDL_aligned_alloc(SIMD_MEMORY_ALIGNMENT, PARTICLES_LENGTH);
		(*vel_y)  = SDL_aligned_alloc(SIMD_MEMORY_ALIGNMENT, PARTICLES_LENGTH);
		if ((*vel_x) == NULL || (*vel_y) == NULL) {
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
#if USE_MULTITHREADING
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
#endif

	return true;
}

static bool initialize_threads(struct particle_system* system)
{

#if USE_MULTITHREADING
	system->work_start = SDL_CreateSemaphore(0);
	system->work_done = SDL_CreateSemaphore(0);

	if (system->work_start == NULL || system->work_done == NULL) {
		SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to create SDL semaphores!\n%s", SDL_GetError());
		return false;
	}

	// Allow threads to start
	SDL_SetAtomicInt(&system->exit_flag, 0);

	size_t st = 0;
	size_t nd = system->pairs_length/system->num_threads;
#if USE_SIMD
	size_t st2 = 0;
	size_t nd2 = system->pairs_simd_length/system->num_threads;
#endif
	for (size_t i = 0; i < system->num_threads; ++i) {
#if USE_SIMD
		const struct thread_data data = {.system = system, .dt = 1e-6f, .thread_id = i, .simd_start = st2, .simd_end = nd2,
			.start = st, .end = nd, .work_start = system->work_start,
			.work_done = system->work_done, .exit_flag = &system->exit_flag,
			.num_updates = 1};
		st += system->pairs_length/system->num_threads;
		nd += system->pairs_length/system->num_threads;
		st2 += system->pairs_simd_length/system->num_threads;
		nd2 += system->pairs_simd_length/system->num_threads;
#else
		const struct thread_data data = {.system = system, .dt = 1e-6f, .thread_id = i, .start = st, .end = nd,
			.simd_start = 0, .simd_end = 0, .work_start = system->work_start,
			.work_done = system->work_done, .exit_flag = &system->exit_flag};
		st += system->pairs_length/system->num_threads;
		nd += system->pairs_length/system->num_threads;
#endif

		system->thread_data[i] = data;
		system->threads[i] = SDL_CreateThread(thread_func, "ParticleWorker", system->thread_data + i);
		if (system->threads[i] == NULL) {
			SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to create a SDL thread!\n%s", SDL_GetError());
		}
	}
#endif

	return true;
}

bool particle_system_init(struct particle_system* system, const uint32_t num_particles)
{
	if (!allocate_memory(system, num_particles)) {
		SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Failed to allocate memory!\n");
		return false;
	}

	pcg32_srandom(&system->rng);

	initialize_particles(system);

	if (!generate_particle_pairs(system)) {
		return false;
	}

	if (!initialize_threads(system)) {
		return false;
	}

	return true;
}

void particle_system_reset(struct particle_system* system) {
	initialize_particles(system);
}

void particle_system_free(struct particle_system* system)
{
#if USE_MULTITHREADING
	SDL_SetAtomicInt(&system->exit_flag, 1);

	// Wake up any threads waiting on work_start
	for (int i = 0; i < system->num_threads; i++) {
		SDL_SignalSemaphore(system->work_start);
	}

	for (size_t i = 0; i < system->num_threads; ++i) {
		SDL_WaitThread(system->threads[i], NULL);
	}

	SDL_DestroySemaphore(system->work_start);
	SDL_DestroySemaphore(system->work_done);

	for (size_t i = 0; i < system->num_threads; ++i) {
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

static void move_particles(struct particle_system *system, const float dt)
{
#if USE_SIMD
#if defined(__AVX512F__) && USE_AVX512
	const float* vel_x = system->vel_x;
	const float* vel_y = system->vel_y;
	float* pos_x = system->pos_x;
	float* pos_y = system->pos_y;

	const __m512 dt_f = _mm512_set1_ps(dt);
	for (size_t i = 0; i < system->num_particles; i += AVX512_FLOATS) {
		const __m512 vel_x_f = _mm512_load_ps(vel_x + i);
		__m512 pos_x_f = _mm512_load_ps(pos_x + i);
		pos_x_f = _mm512_fmadd_ps(dt_f, vel_x_f, pos_x_f);
		_mm512_store_ps(pos_x + i, pos_x_f);

		const __m512 vel_y_f = _mm512_load_ps(vel_y + i);
		__m512 pos_y_f = _mm512_load_ps(pos_y + i);
		pos_y_f = _mm512_fmadd_ps(dt_f, vel_y_f, pos_y_f);
		_mm512_store_ps(pos_y + i, pos_y_f);
	}
#elif defined(__AVX__) && USE_AVX
	const float* vel_x = system->vel_x;
	const float* vel_y = system->vel_y;
	float* pos_x = system->pos_x;
	float* pos_y = system->pos_y;

	const __m256 dt_f = _mm256_set1_ps(dt);
	for (size_t i = 0; i < system->num_particles; i += AVX_FLOATS) {
		const __m256 vel_x_f = _mm256_load_ps(vel_x + i);
		__m256 pos_x_f = _mm256_load_ps(pos_x + i);
#if defined(__FMA__) && USE_FMA
		pos_x_f = _mm256_fmadd_ps(dt_f, vel_x_f, pos_x_f);
#else
		pos_x_f = _mm256_add_ps(_mm256_mul_ps(dt_f, vel_x_f), pos_x_f);
#endif
		_mm256_store_ps(pos_x + i, pos_x_f);

		const __m256 vel_y_f = _mm256_load_ps(vel_y + i);
		__m256 pos_y_f = _mm256_load_ps(pos_y + i);
#if defined(__FMA__) && USE_FMA
		pos_y_f = _mm256_fmadd_ps(dt_f, vel_y_f, pos_y_f);
#else
		pos_y_f = _mm256_add_ps(_mm256_mul_ps(dt_f, vel_y_f), pos_y_f);
#endif
		_mm256_store_ps(pos_y + i, pos_y_f);
	}
#elif defined(__SSE__)
	const float* vel_x = system->vel_x;
	const float* vel_y = system->vel_y;
	float* pos_x = system->pos_x;
	float* pos_y = system->pos_y;

	const __m128 dt_f = _mm_set1_ps(dt);
	for (size_t i = 0; i < system->num_particles; i += SSE_FLOATS) {
		const __m128 vel_x_f = _mm_load_ps(vel_x + i);
		__m128 pos_x_f = _mm_load_ps(pos_x + i);
#if defined(__FMA__) && USE_FMA
		pos_x_f = _mm_fmadd_ps(dt_f, vel_x_f, pos_x_f);
#else
		pos_x_f = _mm_add_ps(_mm_mul_ps(dt_f, vel_x_f), pos_x_f);
#endif
		_mm_store_ps(pos_x + i, pos_x_f);

		const __m128 vel_y_f = _mm_load_ps(vel_y + i);
		__m128 pos_y_f = _mm_load_ps(pos_y + i);
#if defined(__FMA__) && USE_FMA
		pos_y_f = _mm_fmadd_ps(dt_f, vel_y_f, pos_y_f);
#else
		pos_y_f = _mm_add_ps(_mm_mul_ps(dt_f, vel_y_f), pos_y_f);
#endif
		_mm_store_ps(pos_y + i, pos_y_f);
	}
#else
#error SIMD not supported
#endif
#else
	for (size_t i = 0; i < system->num_particles; ++i) {
		system->pos_x[i] += dt * system->vel_x[i];
		system->pos_y[i] += dt * system->vel_y[i];
	}
#endif

}

static void attract_particles(struct particle_system *system, const float dt) {
	const float min_dist = 8*dt;
	for (size_t i = 0; i < system->num_particles; ++i) {
		for (size_t j = i+1; j < system->num_particles; ++j) {
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

static void attract_particles_batched(const struct thread_data* thread_data)
{
	const float dt = thread_data->dt;
	const float min_dist = 8*dt;

	const float* pos_x = thread_data->system->pos_x;
	const float* pos_y = thread_data->system->pos_y;
	float* vel_x = thread_data->system->buffer.vel_x[thread_data->thread_id];
	float* vel_y = thread_data->system->buffer.vel_y[thread_data->thread_id];
	const float* mass = thread_data->system->mass;

	const struct particle_pair* pairs = thread_data->system->pairs;

	for (size_t n = thread_data->start; n < thread_data->end; ++n) {
		const struct particle_pair pair = pairs[n];
		const size_t i = pair.i;
		const size_t j = pair.j;

		const float pos_x_diff = pos_x[i] - pos_x[j];
		const float pos_y_diff = pos_y[i] - pos_y[j];
		const float distance_squared = pos_x_diff * pos_x_diff + pos_y_diff * pos_y_diff;
		const float distance = SDL_max(min_dist, SDL_sqrtf(distance_squared));
		const float force = (mass[i] * mass[j]) / (distance * distance * distance);
		const float force_x = force * pos_x_diff;
		const float force_y = force * pos_y_diff;

		vel_x[i] += dt * force_x / mass[i];
		vel_y[i] += dt * force_y / mass[i];

		vel_x[j] -= dt * force_x / mass[j];
		vel_y[j] -= dt * force_y / mass[j];
	}
}

static void attract_particles_sse(struct particle_system *system, const float dt) {
	const float min_dist = 8*dt;
	const __m128 min_inv_dist_f = _mm_set1_ps(1.0f / min_dist);
	const __m128 dt_f = _mm_set1_ps(dt);

	const int use_particles = SSE_FLOATS*(system->num_particles/SSE_FLOATS);

	const float* pos_x = system->pos_x;
	const float* pos_y = system->pos_y;
	float* vel_x = system->vel_x;
	float* vel_y = system->vel_y;
	const float* mass = system->mass;

	for (size_t hop = SSE_FLOATS; hop < use_particles; ++hop) {
		for (size_t n = 0; n < use_particles/SSE_FLOATS+1; ++n) {
			const size_t i = n*SSE_FLOATS;
			const size_t j = (i + hop) % (use_particles);

			const __m128 pos_x_i = _mm_load_ps(pos_x + i);
			const __m128 pos_y_i = _mm_load_ps(pos_y + i);
			__m128 vel_x_i = _mm_load_ps(vel_x + i);
			__m128 vel_y_i = _mm_load_ps(vel_y + i);
			const __m128 mass_i = _mm_load_ps(mass + i);

			const __m128 pos_x_j = _mm_load_ps(pos_x + j);
			const __m128 pos_y_j = _mm_load_ps(pos_y + j);
			__m128 vel_x_j = _mm_load_ps(vel_x + j);
			__m128 vel_y_j = _mm_load_ps(vel_y + j);
			const __m128 mass_j = _mm_load_ps(mass + j);

			const __m128 pos_x_diff = _mm_sub_ps(pos_x_i, pos_x_j);
			const __m128 pos_y_diff = _mm_sub_ps(pos_y_i, pos_y_j);
			const __m128 distance_squared = _mm_add_ps(_mm_mul_ps(pos_x_diff, pos_x_diff), _mm_mul_ps(pos_y_diff, pos_y_diff));
			const __m128 distance = _mm_min_ps(min_inv_dist_f, _mm_rsqrt_ps(distance_squared));
			const __m128 distance_cubed = _mm_mul_ps(distance, _mm_mul_ps(distance, distance));
			const __m128 force = _mm_mul_ps(_mm_mul_ps(mass_i, mass_j), distance_cubed);
			const __m128 force_x = _mm_mul_ps(force, pos_x_diff);
			const __m128 force_y = _mm_mul_ps(force, pos_y_diff);

#if defined(__FMA__) && USE_FMA
			vel_x_i = _mm_fmadd_ps(dt_f, _mm_div_ps(force_x, mass_i), vel_x_i);
			vel_y_i = _mm_fmadd_ps(dt_f, _mm_div_ps(force_y, mass_i), vel_y_i);

			vel_x_j = _mm_fnmadd_ps(dt_f, _mm_div_ps(force_x, mass_j), vel_x_j);
			vel_y_j = _mm_fnmadd_ps(dt_f, _mm_div_ps(force_y, mass_j), vel_y_j);
#else
			vel_x_i = _mm_add_ps(vel_x_i, _mm_mul_ps(dt_f, _mm_div_ps(force_x, mass_i)));
			vel_y_i = _mm_add_ps(vel_y_i, _mm_mul_ps(dt_f, _mm_div_ps(force_y, mass_i)));

			vel_x_j = _mm_sub_ps(vel_x_j, _mm_mul_ps(dt_f, _mm_div_ps(force_x, mass_j)));
			vel_y_j = _mm_sub_ps(vel_y_j, _mm_mul_ps(dt_f, _mm_div_ps(force_y, mass_j)));
#endif

			_mm_store_ps(vel_x + i, vel_x_i);
			_mm_store_ps(vel_y + i, vel_y_i);
			_mm_store_ps(vel_x + j, vel_x_j);
			_mm_store_ps(vel_y + j, vel_y_j);
		}
	}

	for (size_t k = 0; k < SSE_FLOATS; ++k) {
		for (size_t n = 0; n < system->num_particles; ++n) {
			const size_t i = n;
			const size_t j = (n + k) % system->num_particles;
			const float pos_x_diff = pos_x[i] - pos_x[j];
			const float pos_y_diff = pos_y[i] - pos_y[j];
			const float distance_squared = pos_x_diff * pos_x_diff + pos_y_diff * pos_y_diff;
			const float distance = SDL_max(min_dist, SDL_sqrtf(distance_squared));
			const float force = (mass[i] * mass[j]) / (distance * distance * distance);
			const float force_x = force * pos_x_diff;
			const float force_y = force * pos_y_diff;

			vel_x[i] += dt * force_x / mass[i];
			vel_y[i] += dt * force_y / mass[i];

			vel_x[j] -= dt * force_x / mass[j];
			vel_y[j] -= dt * force_y / mass[j];
		}
	}
}

static void attract_particles_sse_batched(const struct thread_data* thread_data)
{
	const float min_dist = 8*thread_data->dt;
	const __m128 min_inv_dist_f = _mm_set1_ps(1.0f / min_dist);

	const float dt = thread_data->dt;
	const __m128 dt_f = _mm_set1_ps(thread_data->dt);

	const float* pos_x = thread_data->system->pos_x;
	const float* pos_y = thread_data->system->pos_y;
	const float* mass = thread_data->system->mass;
	float* vel_x = thread_data->system->buffer.vel_x[thread_data->thread_id];
	float* vel_y = thread_data->system->buffer.vel_y[thread_data->thread_id];

	const struct particle_pair* pairs_simd = thread_data->system->pairs_simd;
	const struct particle_pair* pairs = thread_data->system->pairs;

	for (size_t k = thread_data->simd_start; k < thread_data->simd_end; ++k) {
		const struct particle_pair pair = pairs_simd[k];
		const size_t i = pair.i;
		const size_t j = pair.j;

		const __m128 pos_x_i = _mm_load_ps(pos_x + i);
		const __m128 pos_y_i = _mm_load_ps(pos_y + i);
		__m128 vel_x_i = _mm_load_ps(vel_x + i);
		__m128 vel_y_i = _mm_load_ps(vel_y + i);
		const __m128 mass_i = _mm_load_ps(mass + i);

		const __m128 pos_x_j = _mm_load_ps(pos_x + j);
		const __m128 pos_y_j = _mm_load_ps(pos_y + j);
		__m128 vel_x_j = _mm_load_ps(vel_x + j);
		__m128 vel_y_j = _mm_load_ps(vel_y + j);
		const __m128 mass_j = _mm_load_ps(mass + j);

		const __m128 pos_x_diff = _mm_sub_ps(pos_x_i, pos_x_j);
		const __m128 pos_y_diff = _mm_sub_ps(pos_y_i, pos_y_j);
		const __m128 distance_squared = _mm_add_ps(_mm_mul_ps(pos_x_diff, pos_x_diff), _mm_mul_ps(pos_y_diff, pos_y_diff));
		const __m128 distance = _mm_min_ps(min_inv_dist_f, _mm_rsqrt_ps(distance_squared));
		const __m128 distance_cubed = _mm_mul_ps(distance, _mm_mul_ps(distance, distance));
		const __m128 force = _mm_mul_ps(_mm_mul_ps(mass_i, mass_j), distance_cubed);
		const __m128 force_x = _mm_mul_ps(force, pos_x_diff);
		const __m128 force_y = _mm_mul_ps(force, pos_y_diff);

#if defined(__FMA__) && USE_FMA
		vel_x_i = _mm_fmadd_ps(dt_f, _mm_div_ps(force_x, mass_i), vel_x_i);
		vel_y_i = _mm_fmadd_ps(dt_f, _mm_div_ps(force_y, mass_i), vel_y_i);

		vel_x_j = _mm_fnmadd_ps(dt_f, _mm_div_ps(force_x, mass_j), vel_x_j);
		vel_y_j = _mm_fnmadd_ps(dt_f, _mm_div_ps(force_y, mass_j), vel_y_j);
#else
		vel_x_i = _mm_add_ps(vel_x_i, _mm_mul_ps(dt_f, _mm_div_ps(force_x, mass_i)));
		vel_y_i = _mm_add_ps(vel_y_i, _mm_mul_ps(dt_f, _mm_div_ps(force_y, mass_i)));

		vel_x_j = _mm_sub_ps(vel_x_j, _mm_mul_ps(dt_f, _mm_div_ps(force_x, mass_j)));
		vel_y_j = _mm_sub_ps(vel_y_j, _mm_mul_ps(dt_f, _mm_div_ps(force_y, mass_j)));
#endif

		_mm_store_ps(vel_x + i, vel_x_i);
		_mm_store_ps(vel_y + i, vel_y_i);
		_mm_store_ps(vel_x + j, vel_x_j);
		_mm_store_ps(vel_y + j, vel_y_j);
	}

	for (size_t k = thread_data->start; k < thread_data->end; ++k) {
		const struct particle_pair pair = pairs[k];
		const size_t i = pair.i;
		const size_t j = pair.j;

		const float pos_x_diff = pos_x[i] - pos_x[j];
		const float pos_y_diff = pos_y[i] - pos_y[j];
		const float distance_squared = pos_x_diff * pos_x_diff + pos_y_diff * pos_y_diff;
		const float distance = SDL_max(min_dist, SDL_sqrtf(distance_squared));
		const float force = (mass[i] * mass[j]) / (distance * distance * distance);
		const float force_x = force * pos_x_diff;
		const float force_y = force * pos_y_diff;

		vel_x[i] += dt * force_x / mass[i];
		vel_y[i] += dt * force_y / mass[i];

		vel_x[j] -= dt * force_x / mass[j];
		vel_y[j] -= dt * force_y / mass[j];
	}
}

static void attract_particles_avx(struct particle_system *system, const float dt) {
	const float min_dist = 8*dt;
	const __m256 min_inv_dist_f = _mm256_set1_ps(1.0f / min_dist);
	const __m256 dt_f = _mm256_set1_ps(dt);

	const int use_particles = AVX_FLOATS*(system->num_particles/AVX_FLOATS);

	const float* pos_x = system->pos_x;
	const float* pos_y = system->pos_y;
	float* vel_x = system->vel_x;
	float* vel_y = system->vel_y;
	const float* mass = system->mass;

	for (size_t hop = AVX_FLOATS; hop < use_particles; ++hop) {
		for (size_t n = 0; n < use_particles/AVX_FLOATS+1; ++n) {
			const size_t i = n*AVX_FLOATS;
			const size_t j = (i + hop) % (use_particles);

			const __m256 pos_x_i = _mm256_load_ps(pos_x + i);
			const __m256 pos_y_i = _mm256_load_ps(pos_y + i);
			__m256 vel_x_i = _mm256_load_ps(vel_x + i);
			__m256 vel_y_i = _mm256_load_ps(vel_y + i);
			const __m256 mass_i = _mm256_load_ps(mass + i);

			const __m256 pos_x_j = _mm256_load_ps(pos_x + j);
			const __m256 pos_y_j = _mm256_load_ps(pos_y + j);
			__m256 vel_x_j = _mm256_load_ps(vel_x + j);
			__m256 vel_y_j = _mm256_load_ps(vel_y + j);
			const __m256 mass_j = _mm256_load_ps(mass + j);

			const __m256 pos_x_diff = _mm256_sub_ps(pos_x_i, pos_x_j);
			const __m256 pos_y_diff = _mm256_sub_ps(pos_y_i, pos_y_j);
			const __m256 distance_squared = _mm256_add_ps(_mm256_mul_ps(pos_x_diff, pos_x_diff), _mm256_mul_ps(pos_y_diff, pos_y_diff));
			const __m256 distance = _mm256_min_ps(min_inv_dist_f, _mm256_rsqrt_ps(distance_squared));
			const __m256 distance_cubed = _mm256_mul_ps(distance, _mm256_mul_ps(distance, distance));
			const __m256 force = _mm256_mul_ps(_mm256_mul_ps(mass_i, mass_j), distance_cubed);
			const __m256 force_x = _mm256_mul_ps(force, pos_x_diff);
			const __m256 force_y = _mm256_mul_ps(force, pos_y_diff);

#if defined(__FMA__) && USE_FMA
			vel_x_i = _mm256_fmadd_ps(dt_f, _mm256_div_ps(force_x, mass_i), vel_x_i);
			vel_y_i = _mm256_fmadd_ps(dt_f, _mm256_div_ps(force_y, mass_i), vel_y_i);

			vel_x_j = _mm256_fnmadd_ps(dt_f, _mm256_div_ps(force_x, mass_j), vel_x_j);
			vel_y_j = _mm256_fnmadd_ps(dt_f, _mm256_div_ps(force_y, mass_j), vel_y_j);
#else
			vel_x_i = _mm256_add_ps(vel_x_i, _mm256_mul_ps(dt_f, _mm256_div_ps(force_x, mass_i)));
			vel_y_i = _mm256_add_ps(vel_y_i, _mm256_mul_ps(dt_f, _mm256_div_ps(force_y, mass_i)));

			vel_x_j = _mm256_sub_ps(vel_x_j, _mm256_mul_ps(dt_f, _mm256_div_ps(force_x, mass_j)));
			vel_y_j = _mm256_sub_ps(vel_y_j, _mm256_mul_ps(dt_f, _mm256_div_ps(force_y, mass_j)));
#endif

			_mm256_store_ps(vel_x + i, vel_x_i);
			_mm256_store_ps(vel_y + i, vel_y_i);
			_mm256_store_ps(vel_x + j, vel_x_j);
			_mm256_store_ps(vel_y + j, vel_y_j);
		}
	}

	for (size_t k = 0; k < AVX_FLOATS; ++k) {
		for (size_t n = 0; n < system->num_particles; ++n) {
			const size_t i = n;
			const size_t j = (n + k) % system->num_particles;
			const float pos_x_diff = pos_x[i] - pos_x[j];
			const float pos_y_diff = pos_y[i] - pos_y[j];
			const float distance_squared = pos_x_diff * pos_x_diff + pos_y_diff * pos_y_diff;
			const float distance = SDL_max(min_dist, SDL_sqrtf(distance_squared));
			const float force = (mass[i] * mass[j]) / (distance * distance * distance);
			const float force_x = force * pos_x_diff;
			const float force_y = force * pos_y_diff;

			vel_x[i] += dt * force_x / mass[i];
			vel_y[i] += dt * force_y / mass[i];

			vel_x[j] -= dt * force_x / mass[j];
			vel_y[j] -= dt * force_y / mass[j];
		}
	}
}

static void attract_particles_avx_batched(const struct thread_data* thread_data)
{
	const float min_dist = 8*thread_data->dt;
	const __m256 min_inv_dist_f = _mm256_set1_ps(1.0f / min_dist);

	const float dt = thread_data->dt;
	const __m256 dt_f = _mm256_set1_ps(dt);

	const float* pos_x = thread_data->system->pos_x;
	const float* pos_y = thread_data->system->pos_y;
	const float* mass = thread_data->system->mass;
	float* vel_x = thread_data->system->buffer.vel_x[thread_data->thread_id];
	float* vel_y = thread_data->system->buffer.vel_y[thread_data->thread_id];

	const struct particle_pair* pairs_simd = thread_data->system->pairs_simd;
	const struct particle_pair* pairs = thread_data->system->pairs;

	for (size_t k = thread_data->simd_start; k < thread_data->simd_end; ++k) {
		const struct particle_pair pair = pairs_simd[k];
		const size_t i = pair.i;
		const size_t j = pair.j;

		const __m256 pos_x_i = _mm256_load_ps(pos_x + i);
		const __m256 pos_y_i = _mm256_load_ps(pos_y + i);
		__m256 vel_x_i = _mm256_load_ps(vel_x + i);
		__m256 vel_y_i = _mm256_load_ps(vel_y + i);
		const __m256 mass_i = _mm256_load_ps(mass + i);

		const __m256 pos_x_j = _mm256_load_ps(pos_x + j);
		const __m256 pos_y_j = _mm256_load_ps(pos_y + j);
		__m256 vel_x_j = _mm256_load_ps(vel_x + j);
		 __m256 vel_y_j = _mm256_load_ps(vel_y + j);
		const __m256 mass_j = _mm256_load_ps(mass + j);

		const __m256 pos_x_diff = _mm256_sub_ps(pos_x_i, pos_x_j);
		const __m256 pos_y_diff = _mm256_sub_ps(pos_y_i, pos_y_j);
		const __m256 distance_squared = _mm256_add_ps(_mm256_mul_ps(pos_x_diff, pos_x_diff), _mm256_mul_ps(pos_y_diff, pos_y_diff));
		const __m256 distance = _mm256_min_ps(min_inv_dist_f, _mm256_rsqrt_ps(distance_squared));
		const __m256 distance_cubed = _mm256_mul_ps(distance, _mm256_mul_ps(distance, distance));
		const __m256 force = _mm256_mul_ps(_mm256_mul_ps(mass_i, mass_j), distance_cubed);
		const __m256 force_x = _mm256_mul_ps(force, pos_x_diff);
		const __m256 force_y = _mm256_mul_ps(force, pos_y_diff);

#if defined(__FMA__) && USE_FMA
		vel_x_i = _mm256_fmadd_ps(dt_f, _mm256_div_ps(force_x, mass_i), vel_x_i);
		vel_y_i = _mm256_fmadd_ps(dt_f, _mm256_div_ps(force_y, mass_i), vel_y_i);

		vel_x_j = _mm256_fnmadd_ps(dt_f, _mm256_div_ps(force_x, mass_j), vel_x_j);
		vel_y_j = _mm256_fnmadd_ps(dt_f, _mm256_div_ps(force_y, mass_j), vel_y_j);
#else
		vel_x_i = _mm256_add_ps(vel_x_i, _mm256_mul_ps(dt_f, _mm256_div_ps(force_x, mass_i)));
		vel_y_i = _mm256_add_ps(vel_y_i, _mm256_mul_ps(dt_f, _mm256_div_ps(force_y, mass_i)));

		vel_x_j = _mm256_sub_ps(vel_x_j, _mm256_mul_ps(dt_f, _mm256_div_ps(force_x, mass_j)));
		vel_y_j = _mm256_sub_ps(vel_y_j, _mm256_mul_ps(dt_f, _mm256_div_ps(force_y, mass_j)));
#endif

		_mm256_store_ps(vel_x + i, vel_x_i);
		_mm256_store_ps(vel_y + i, vel_y_i);
		_mm256_store_ps(vel_x + j, vel_x_j);
		_mm256_store_ps(vel_y + j, vel_y_j);
	}

	for (size_t k = thread_data->start; k < thread_data->end; ++k) {
		const struct particle_pair pair = pairs[k];
		const size_t i = pair.i;
		const size_t j = pair.j;

		const float pos_x_diff = pos_x[i] - pos_x[j];
		const float pos_y_diff = pos_y[i] - pos_y[j];
		const float distance_squared = pos_x_diff * pos_x_diff + pos_y_diff * pos_y_diff;
		const float distance = SDL_max(min_dist, SDL_sqrtf(distance_squared));
		const float force = (mass[i] * mass[j]) / (distance * distance * distance);
		const float force_x = force * pos_x_diff;
		const float force_y = force * pos_y_diff;

		vel_x[i] += dt * force_x / mass[i];
		vel_y[i] += dt * force_y / mass[i];

		vel_x[j] -= dt * force_x / mass[j];
		vel_y[j] -= dt * force_y / mass[j];
	}
}

static void attract_particles_avx512(struct particle_system *system, const float dt) {
	const float min_dist = 8*dt;
	const __m512 min_inv_dist_f = _mm512_set1_ps(1.0f / min_dist);
	const __m512 dt_f = _mm512_set1_ps(dt);

	const int use_particles = AVX512_FLOATS*(system->num_particles/AVX512_FLOATS);

	const float* pos_x = system->pos_x;
	const float* pos_y = system->pos_y;
	float* vel_x = system->vel_x;
	float* vel_y = system->vel_y;
	const float* mass = system->mass;

	for (size_t hop = AVX512_FLOATS; hop < use_particles; ++hop) {
		for (size_t n = 0; n < use_particles/AVX512_FLOATS+1; ++n) {
			const size_t i = n*AVX512_FLOATS;
			const size_t j = (i + hop) % (use_particles);

			const __m512 pos_x_i = _mm512_load_ps(pos_x + i);
			const __m512 pos_y_i = _mm512_load_ps(pos_y + i);
			__m512 vel_x_i = _mm512_load_ps(vel_x + i);
			__m512 vel_y_i = _mm512_load_ps(vel_y + i);
			const __m512 mass_i = _mm512_load_ps(mass + i);

			const __m512 pos_x_j = _mm512_load_ps(pos_x + j);
			const __m512 pos_y_j = _mm512_load_ps(pos_y + j);
			__m512 vel_x_j = _mm512_load_ps(vel_x + j);
			__m512 vel_y_j = _mm512_load_ps(vel_y + j);
			const __m512 mass_j = _mm512_load_ps(mass + j);

			const __m512 pos_x_diff = _mm512_sub_ps(pos_x_i, pos_x_j);
			const __m512 pos_y_diff = _mm512_sub_ps(pos_y_i, pos_y_j);
			const __m512 distance_squared = _mm512_add_ps(_mm512_mul_ps(pos_x_diff, pos_x_diff), _mm512_mul_ps(pos_y_diff, pos_y_diff));
			const __m512 distance = _mm512_min_ps(min_inv_dist_f, _mm512_rsqrt14_ps(distance_squared));
			const __m512 distance_cubed = _mm512_mul_ps(distance, _mm512_mul_ps(distance, distance));
			const __m512 force = _mm512_mul_ps(_mm512_mul_ps(mass_i, mass_j), distance_cubed);
			const __m512 force_x = _mm512_mul_ps(force, pos_x_diff);
			const __m512 force_y = _mm512_mul_ps(force, pos_y_diff);

			vel_x_i = _mm512_fmadd_ps(dt_f, _mm512_div_ps(force_x, mass_i), vel_x_i);
			vel_y_i = _mm512_fmadd_ps(dt_f, _mm512_div_ps(force_y, mass_i), vel_y_i);

			vel_x_j = _mm512_fnmadd_ps(dt_f, _mm512_div_ps(force_x, mass_j), vel_x_j);
			vel_y_j = _mm512_fnmadd_ps(dt_f, _mm512_div_ps(force_y, mass_j), vel_y_j);

			_mm512_store_ps(vel_x + i, vel_x_i);
			_mm512_store_ps(vel_y + i, vel_y_i);
			_mm512_store_ps(vel_x + j, vel_x_j);
			_mm512_store_ps(vel_y + j, vel_y_j);
		}
	}

	for (size_t k = 0; k < AVX512_FLOATS; ++k) {
		for (size_t n = 0; n < system->num_particles; ++n) {
			const size_t i = n;
			const size_t j = (n + k) % system->num_particles;
			const float pos_x_diff = pos_x[i] - pos_x[j];
			const float pos_y_diff = pos_y[i] - pos_y[j];
			const float distance_squared = pos_x_diff * pos_x_diff + pos_y_diff * pos_y_diff;
			const float distance = SDL_max(min_dist, SDL_sqrtf(distance_squared));
			const float force = (mass[i] * mass[j]) / (distance * distance * distance);
			const float force_x = force * pos_x_diff;
			const float force_y = force * pos_y_diff;

			vel_x[i] += dt * force_x / mass[i];
			vel_y[i] += dt * force_y / mass[i];

			vel_x[j] -= dt * force_x / mass[j];
			vel_y[j] -= dt * force_y / mass[j];
		}
	}
}

static void attract_particles_avx512_batched(const struct thread_data* thread_data) {
	const float min_dist = 8*thread_data->dt;
	const __m512 min_inv_dist_f = _mm512_set1_ps(1.0f / min_dist);

	const float dt = thread_data->dt;
	const __m512 dt_f = _mm512_set1_ps(dt);

	const float* pos_x = thread_data->system->pos_x;
	const float* pos_y = thread_data->system->pos_y;
	const float* mass = thread_data->system->mass;
	float* vel_x = thread_data->system->buffer.vel_x[thread_data->thread_id];
	float* vel_y = thread_data->system->buffer.vel_y[thread_data->thread_id];

	const struct particle_pair* pairs_simd = thread_data->system->pairs_simd;
	const struct particle_pair* pairs = thread_data->system->pairs;

	for (size_t k = thread_data->simd_start; k < thread_data->simd_end; ++k) {
		const struct particle_pair pair = pairs_simd[k];
		const size_t i = pair.i;
		const size_t j = pair.j;

		const __m512 pos_x_i = _mm512_load_ps(pos_x + i);
		const __m512 pos_y_i = _mm512_load_ps(pos_y + i);
		__m512 vel_x_i = _mm512_load_ps(vel_x + i);
		__m512 vel_y_i = _mm512_load_ps(vel_y + i);
		const __m512 mass_i = _mm512_load_ps(mass + i);

		const __m512 pos_x_j = _mm512_load_ps(pos_x + j);
		const __m512 pos_y_j = _mm512_load_ps(pos_y + j);
		__m512 vel_x_j = _mm512_load_ps(vel_x + j);
		__m512 vel_y_j = _mm512_load_ps(vel_y + j);
		const __m512 mass_j = _mm512_load_ps(mass + j);

		const __m512 pos_x_diff = _mm512_sub_ps(pos_x_i, pos_x_j);
		const __m512 pos_y_diff = _mm512_sub_ps(pos_y_i, pos_y_j);
		const __m512 distance_squared = _mm512_add_ps(_mm512_mul_ps(pos_x_diff, pos_x_diff), _mm512_mul_ps(pos_y_diff, pos_y_diff));
		const __m512 distance = _mm512_min_ps(min_inv_dist_f, _mm512_rsqrt14_ps(distance_squared));
		const __m512 distance_cubed = _mm512_mul_ps(distance, _mm512_mul_ps(distance, distance));
		const __m512 force = _mm512_mul_ps(_mm512_mul_ps(mass_i, mass_j), distance_cubed);
		const __m512 force_x = _mm512_mul_ps(force, pos_x_diff);
		const __m512 force_y = _mm512_mul_ps(force, pos_y_diff);

		vel_x_i = _mm512_fmadd_ps(dt_f, _mm512_div_ps(force_x, mass_i), vel_x_i);
		vel_y_i = _mm512_fmadd_ps(dt_f, _mm512_div_ps(force_y, mass_i), vel_y_i);

		vel_x_j = _mm512_fnmadd_ps(dt_f, _mm512_div_ps(force_x, mass_j), vel_x_j);
		vel_y_j = _mm512_fnmadd_ps(dt_f, _mm512_div_ps(force_y, mass_j), vel_y_j);

		_mm512_store_ps(vel_x + i, vel_x_i);
		_mm512_store_ps(vel_y + i, vel_y_i);
		_mm512_store_ps(vel_x + j, vel_x_j);
		_mm512_store_ps(vel_y + j, vel_y_j);
	}

	for (size_t k = thread_data->start; k < thread_data->end; ++k) {
		const struct particle_pair pair = pairs[k];
		const size_t i = pair.i;
		const size_t j = pair.j;

		const float pos_x_diff = pos_x[i] - pos_x[j];
		const float pos_y_diff = pos_y[i] - pos_y[j];
		const float distance_squared = pos_x_diff * pos_x_diff + pos_y_diff * pos_y_diff;
		const float distance = SDL_max(min_dist, SDL_sqrtf(distance_squared));
		const float force = (mass[i] * mass[j]) / (distance * distance * distance);
		const float force_x = force * pos_x_diff;
		const float force_y = force * pos_y_diff;

		vel_x[i] += dt * force_x / mass[i];
		vel_y[i] += dt * force_y / mass[i];

		vel_x[j] -= dt * force_x / mass[j];
		vel_y[j] -= dt * force_y / mass[j];
	}
}

static void calculate_average(struct particle_system* system, float* out_avg_x, float* out_avg_y)
{
#if USE_SIMD
#if defined(__AVX512F__) && USE_AVX512
	const float* pos_x = system->pos_x;
	const float* pos_y = system->pos_y;

	__m512 avg_x_f = _mm512_set1_ps(0.0f);
	__m512 avg_y_f = _mm512_set1_ps(0.0f);
	for (size_t i = 0; i < system->num_particles; i += AVX512_FLOATS) {
		const __m512 pos_x_f = _mm512_load_ps(pos_x + i);
		avg_x_f = _mm512_add_ps(avg_x_f, pos_x_f);

		const __m512 pos_y_f = _mm512_load_ps(pos_y + i);
		avg_y_f = _mm512_add_ps(avg_y_f, pos_y_f);
	}
	(*out_avg_x) = _mm512_reduce_add_ps(avg_x_f) / (float)system->num_particles;
	(*out_avg_y) = _mm512_reduce_add_ps(avg_y_f) / (float)system->num_particles;
#elif defined(__AVX__) && USE_AVX
	const float* pos_x = system->pos_x;
	const float* pos_y = system->pos_y;

	__m256 avg_x_f = _mm256_set1_ps(0.0f);
	__m256 avg_y_f = _mm256_set1_ps(0.0f);
	for (size_t i = 0; i < system->num_particles; i += AVX_FLOATS) {
		const __m256 pos_x_f = _mm256_load_ps(pos_x + i);
		avg_x_f = _mm256_add_ps(avg_x_f, pos_x_f);

		const __m256 pos_y_f = _mm256_load_ps(pos_y + i);
		avg_y_f = _mm256_add_ps(avg_y_f, pos_y_f);
	}
	(*out_avg_x) = hsum256_ps_avx(avg_x_f) / (float)system->num_particles;
	(*out_avg_y) = hsum256_ps_avx(avg_y_f) / (float)system->num_particles;
#elif defined(__SSE__)
	const float* pos_x = system->pos_x;
	const float* pos_y = system->pos_y;

	__m128 avg_x_f = _mm_set1_ps(0.0f);
	__m128 avg_y_f = _mm_set1_ps(0.0f);
	for (size_t i = 0; i < system->num_particles; i += SSE_FLOATS) {
		const __m128 pos_x_f = _mm_load_ps(pos_x + i);
		avg_x_f = _mm_add_ps(avg_x_f, pos_x_f);

		const __m128 pos_y_f = _mm_load_ps(pos_y + i);
		avg_y_f = _mm_add_ps(avg_y_f, pos_y_f);
	}
	(*out_avg_x) = hsum_ps_sse(avg_x_f) / (float)system->num_particles;
	(*out_avg_y) = hsum_ps_sse(avg_y_f) / (float)system->num_particles;
#else
#error SIMD not supported
#endif
#else
	float avg_x = 0.0f;
	float avg_y = 0.0f;
	for (size_t i = 0; i < system->num_particles; ++i) {
		avg_x += system->pos_x[i];
		avg_y += system->pos_y[i];
	}
	avg_x /= (float)system->num_particles;
	avg_y /= (float)system->num_particles;
	(*out_avg_x) = avg_x;
	(*out_avg_y) = avg_y;
#endif
}

static float calculate_standard_distribution(struct particle_system* system)
{
	float x_avg, y_avg;
	calculate_average(system, &x_avg, &y_avg);

#if USE_SIMD
#if defined(__AVX512F__) && USE_AVX512
	const float* pos_x = system->pos_x;
	const float* pos_y = system->pos_y;

	__m512 x_dist_f = _mm512_set1_ps(0.0f);
	__m512 y_dist_f = _mm512_set1_ps(0.0f);
	const __m512 x_avg_f = _mm512_set1_ps(x_avg);
	const __m512 y_avg_f = _mm512_set1_ps(y_avg);
	for (size_t i = 0; i < system->num_particles; i += AVX512_FLOATS) {
		const __m512 pos_x_f = _mm512_load_ps(pos_x + i);
		const __m512 diff_x = _mm512_sub_ps(pos_x_f, x_avg_f);
		x_dist_f = _mm512_fmadd_ps(diff_x, diff_x, x_dist_f);

		const __m512 pos_y_f = _mm512_load_ps(pos_y + i);
		const __m512 diff_y = _mm512_sub_ps(pos_y_f, y_avg_f);
		y_dist_f = _mm512_fmadd_ps(diff_y, diff_y, y_dist_f);
	}
	const float x_dist = _mm512_reduce_add_ps(x_dist_f) / (float)system->num_particles;
	const float y_dist = _mm512_reduce_add_ps(y_dist_f) / (float)system->num_particles;
	return SDL_sqrtf(x_dist + y_dist);
#elif defined(__AVX__) && USE_AVX
	const float* pos_x = system->pos_x;
	const float* pos_y = system->pos_y;

	__m256 x_dist_f = _mm256_set1_ps(0.0f);
	__m256 y_dist_f = _mm256_set1_ps(0.0f);
	const __m256 x_avg_f = _mm256_set1_ps(x_avg);
	const __m256 y_avg_f = _mm256_set1_ps(y_avg);
	for (size_t i = 0; i < system->num_particles; i += AVX_FLOATS) {
		const __m256 pos_x_f = _mm256_load_ps(pos_x + i);
		const __m256 diff_x = _mm256_sub_ps(pos_x_f, x_avg_f);
#if defined(__FMA__) && USE_FMA
		x_dist_f = _mm256_fmadd_ps(diff_x, diff_x, x_dist_f);
#else
		x_dist_f = _mm256_add_ps(_mm256_mul_ps(diff_x, diff_x), x_dist_f);
#endif

		const __m256 pos_y_f = _mm256_load_ps(pos_y + i);
		const __m256 diff_y = _mm256_sub_ps(pos_y_f, y_avg_f);
#if defined(__FMA__) && USE_FMA
		y_dist_f = _mm256_fmadd_ps(diff_y, diff_y, y_dist_f);
#else
		y_dist_f = _mm256_add_ps(_mm256_mul_ps(diff_y, diff_y), y_dist_f);
#endif
	}
	const float x_dist = hsum256_ps_avx(x_dist_f) / (float)system->num_particles;
	const float y_dist = hsum256_ps_avx(y_dist_f) / (float)system->num_particles;
	return SDL_sqrtf(x_dist + y_dist);
#elif defined(__SSE__)
	const float* pos_x = system->pos_x;
	const float* pos_y = system->pos_y;

	__m128 x_dist_f = _mm_set1_ps(0.0f);
	__m128 y_dist_f = _mm_set1_ps(0.0f);
	const __m128 x_avg_f = _mm_set1_ps(x_avg);
	const __m128 y_avg_f = _mm_set1_ps(y_avg);
	for (size_t i = 0; i < system->num_particles; i += SSE_FLOATS) {
		const __m128 pos_x_f = _mm_load_ps(pos_x + i);
		const __m128 diff_x = _mm_sub_ps(pos_x_f, x_avg_f);
#if defined(__FMA__) && USE_FMA
		x_dist_f = _mm_fmadd_ps(diff_x, diff_x, x_dist_f);
#else
		x_dist_f = _mm_add_ps(_mm_mul_ps(diff_x, diff_x), x_dist_f);
#endif

		const __m128 pos_y_f = _mm_load_ps(pos_y + i);
		const __m128 diff_y = _mm_sub_ps(pos_y_f, y_avg_f);
#if defined(__FMA__) && USE_FMA
		y_dist_f = _mm_fmadd_ps(diff_y, diff_y, y_dist_f);
#else
		y_dist_f = _mm_add_ps(_mm_mul_ps(diff_y, diff_y), y_dist_f);
#endif
	}
	const float x_dist = hsum_ps_sse(x_dist_f) / (float)system->num_particles;
	const float y_dist = hsum_ps_sse(y_dist_f) / (float)system->num_particles;
	return SDL_sqrtf(x_dist + y_dist);
#else
#error SIMD not supported
#endif
#else
	float x_dist = 0.0f;
	float y_dist = 0.0f;
	for (size_t i = 0; i < system->num_particles; ++i) {
		x_dist += (system->pos_x[i] - x_avg)*(system->pos_x[i] - x_avg);
		y_dist += (system->pos_y[i] - y_avg)*(system->pos_y[i] - y_avg);
	}
	x_dist /= (float)system->num_particles;
	y_dist /= (float)system->num_particles;
	return SDL_sqrtf(x_dist + y_dist);
#endif
}

static void expand_universe(struct particle_system* system, const float amount)
{
	float avg_x, avg_y;
	calculate_average(system, &avg_x, &avg_y);

#if USE_SIMD
#if defined(__AVX512F__) && USE_AVX512
	float* pos_x = system->pos_x;
	float* pos_y = system->pos_y;

	const __m512 amount_f = _mm512_set1_ps(amount);
	const __m512 avg_x_f = _mm512_set1_ps(avg_x);
	const __m512 avg_y_f = _mm512_set1_ps(avg_y);
	for (size_t i = 0; i < system->num_particles; i += AVX512_FLOATS) {
		__m512 pos_x_f = _mm512_load_ps(pos_x + i);
		pos_x_f = _mm512_mul_ps(amount_f, _mm512_sub_ps(pos_x_f, avg_x_f));
		_mm512_store_ps(pos_x + i, pos_x_f);

		__m512 pos_y_f = _mm512_load_ps(pos_y + i);
		pos_y_f = _mm512_mul_ps(amount_f, _mm512_sub_ps(pos_y_f, avg_y_f));
		_mm512_store_ps(pos_y + i, pos_y_f);
	}
#elif defined(__AVX__) && USE_AVX
	float* pos_x = system->pos_x;
	float* pos_y = system->pos_y;

	const __m256 amount_f = _mm256_set1_ps(amount);
	const __m256 avg_x_f = _mm256_set1_ps(avg_x);
	const __m256 avg_y_f = _mm256_set1_ps(avg_y);
	for (size_t i = 0; i < system->num_particles; i += AVX_FLOATS) {
		__m256 pos_x_f = _mm256_load_ps(pos_x + i);
		pos_x_f = _mm256_mul_ps(amount_f, _mm256_sub_ps(pos_x_f, avg_x_f));
		_mm256_store_ps(pos_x + i, pos_x_f);

		__m256 pos_y_f = _mm256_load_ps(pos_y + i);
		pos_y_f = _mm256_mul_ps(amount_f, _mm256_sub_ps(pos_y_f, avg_y_f));
		_mm256_store_ps(pos_y + i, pos_y_f);
	}
#elif defined(__SSE__)
	float* pos_x = system->pos_x;
	float* pos_y = system->pos_y;

	const __m128 amount_f = _mm_set1_ps(amount);
	const __m128 avg_x_f = _mm_set1_ps(avg_x);
	const __m128 avg_y_f = _mm_set1_ps(avg_y);
	for (size_t i = 0; i < system->num_particles; i += SSE_FLOATS) {
		__m128 pos_x_f = _mm_load_ps(pos_x + i);
		pos_x_f = _mm_mul_ps(amount_f, _mm_sub_ps(pos_x_f, avg_x_f));
		_mm_store_ps(pos_x + i, pos_x_f);

		__m128 pos_y_f = _mm_load_ps(pos_y + i);
		pos_y_f = _mm_mul_ps(amount_f, _mm_sub_ps(pos_y_f, avg_y_f));
		_mm_store_ps(pos_y + i, pos_y_f);
	}
#else
#error SIMD not supported
#endif

#else
	for (size_t i = 0; i < system->num_particles; ++i) {
		system->pos_x[i] = amount * (system->pos_x[i] - avg_x);
		system->pos_y[i] = amount * (system->pos_y[i] - avg_y);
	}
#endif
}

void particle_system_update(struct particle_system* system, float dt, uint32_t num_updates)
{
	if (num_updates == 0) {
		return;
	}

	for (size_t thread = 0; thread < system->num_threads; ++thread) {
		system->thread_data[thread].dt = dt;
		system->thread_data[thread].num_updates = num_updates;
	}

	const float amount = 0.7f / calculate_standard_distribution(system);
	expand_universe(system, amount);

#if PARTICLE_SYSTEM_ADD_RANDOM_FORCE
	const float rotation = 0.001f * SDL_powf((float)system->num_particles, 0.666f);
	for (size_t i = 0; i < system->num_particles; ++i) {
		system->vel_x[i] += system->pos_y[i]*rotation + ((float)pcg32_random_r(&system->rng) / (float)SDL_MAX_UINT32 * 2.0f - 1.0f);
		system->vel_y[i] += -system->pos_x[i]*rotation + ((float)pcg32_random_r(&system->rng) / (float)SDL_MAX_UINT32 * 2.0f - 1.0f);
	}
#endif

#if USE_SIMD && !USE_MULTITHREADING
#if defined(__AVX512F__) && USE_AVX512
	for (uint32_t i = 0; i < num_updates; ++i) {
		attract_particles_avx512(system, dt);
	}
#elif defined(__AVX__) && USE_AVX
	for (uint32_t i = 0; i < num_updates; ++i) {
		attract_particles_avx(system, dt);
	}
#elif defined(__SSE__)
	for (uint32_t i = 0; i < num_updates; ++i) {
		attract_particles_sse(system, dt);
	}
#else
#error SIMD not supported
#endif
#else
#if USE_MULTITHREADING
	// Signal all threads to start work
	for (int i = 0; i < system->num_threads; i++) {
		SDL_SignalSemaphore(system->work_start);
	}

	// Wait for all threads to finish
	for (int i = 0; i < system->num_threads; i++) {
		SDL_WaitSemaphore(system->work_done);
	}
#else
	for (uint32_t i = 0; i < num_updates; ++i) {
		attract_particles(system, dt);
	}
#endif

#endif

#if USE_MULTITHREADING
	for (size_t i = 0; i < system->num_particles; ++i) {
		for (size_t thread = 0; thread < system->num_threads; ++thread) {
			system->vel_x[i] += system->buffer.vel_x[thread][i];
			system->vel_y[i] += system->buffer.vel_y[thread][i];
			system->buffer.vel_x[thread][i] = 0.0f;
			system->buffer.vel_y[thread][i] = 0.0f;
		}
	}
#endif

	move_particles(system, dt);
}