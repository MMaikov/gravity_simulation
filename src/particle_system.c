#include "particle_system.h"

#include <SDL.h>

#include "config.h"

#ifdef __AVX512F__
#define SIMD_MEMORY_ALIGNMENT 64
#else
#define SIMD_MEMORY_ALIGNMENT 32
#endif

#include <immintrin.h>
#include "simd_util.h"

static void copy_pos_to_points(struct particle_system* system)
{
	for (size_t i = 0; i < MAX_PARTICLES; ++i) {
		system->points[i].x = (system->pos_x[i] + 1.6f) * (WINDOW_HEIGHT / 2.0f);
		system->points[i].y = (system->pos_y[i] + 1.0f) * (WINDOW_HEIGHT / 2.0f);
	}
}

bool particle_system_init(struct particle_system* system)
{
	const size_t PARTICLES_LENGTH = (MAX_PARTICLES+16) * sizeof(float);

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

	size_t i;
	for (i = 0; i < MAX_PARTICLES+16; ++i) {
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

static void move_particles(struct particle_system *system, const float dt)
{
#if USE_SIMD
#ifdef __AVX512F__
	const float* vel_x = system->vel_x;
	const float* vel_y = system->vel_y;
	float* pos_x = system->pos_x;
	float* pos_y = system->pos_y;

	const __m512 dt_f = _mm512_set1_ps(dt);
	for (size_t i = 0; i < MAX_PARTICLES; i += AVX512_FLOATS) {
		const __m512 vel_x_f = _mm512_load_ps(vel_x + i);
		__m512 pos_x_f = _mm512_load_ps(pos_x + i);
		pos_x_f = _mm512_fmadd_ps(dt_f, vel_x_f, pos_x_f);
		_mm512_store_ps(pos_x + i, pos_x_f);

		const __m512 vel_y_f = _mm512_load_ps(vel_y + i);
		__m512 pos_y_f = _mm512_load_ps(pos_y + i);
		pos_y_f = _mm512_fmadd_ps(dt_f, vel_y_f, pos_y_f);
		_mm512_store_ps(pos_y + i, pos_y_f);
	}
#elif defined __AVX2__
	const float* vel_x = system->vel_x;
	const float* vel_y = system->vel_y;
	float* pos_x = system->pos_x;
	float* pos_y = system->pos_y;

	const __m256 dt_f = _mm256_set1_ps(dt);
	for (size_t i = 0; i < MAX_PARTICLES; i += AVX2_FLOATS) {
		const __m256 vel_x_f = _mm256_load_ps(vel_x + i);
		__m256 pos_x_f = _mm256_load_ps(pos_x + i);
		pos_x_f = _mm256_fmadd_ps(dt_f, vel_x_f, pos_x_f);
		_mm256_store_ps(pos_x + i, pos_x_f);

		const __m256 vel_y_f = _mm256_load_ps(vel_y + i);
		__m256 pos_y_f = _mm256_load_ps(pos_y + i);
		pos_y_f = _mm256_fmadd_ps(dt_f, vel_y_f, pos_y_f);
		_mm256_store_ps(pos_y + i, pos_y_f);
	}
#else
#error SIMD not supported
#endif
#else
	for (size_t i = 0; i < MAX_PARTICLES; ++i) {
		system->pos_x[i] += dt * system->vel_x[i];
		system->pos_y[i] += dt * system->vel_y[i];
	}
#endif

}

static void attract_particles(struct particle_system *system, const float dt) {
	const float min_dist = 8*dt;
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

static void attract_particles_avx2(struct particle_system *system, const float dt) {
	const float min_dist = 8*dt;
	const __m256 min_inv_dist_f = _mm256_set1_ps(1.0f / min_dist);
	const __m256 dt_f = _mm256_set1_ps(dt);

	const int use_particles = AVX2_FLOATS*(MAX_PARTICLES/AVX2_FLOATS);

	for (size_t hop = AVX2_FLOATS; hop < use_particles; ++hop) {
		for (size_t n = 0; n < use_particles/AVX2_FLOATS+1; ++n) {
			const size_t i = n*AVX2_FLOATS;
			const size_t j = (i + hop) % (use_particles);

			__m256 pos_x_i = _mm256_load_ps(system->pos_x + i);
			__m256 pos_y_i = _mm256_load_ps(system->pos_y + i);
			__m256 vel_x_i = _mm256_load_ps(system->vel_x + i);
			__m256 vel_y_i = _mm256_load_ps(system->vel_y + i);
			__m256 mass_i = _mm256_load_ps(system->mass + i);

			__m256 pos_x_j = _mm256_load_ps(system->pos_x + j);
			__m256 pos_y_j = _mm256_load_ps(system->pos_y + j);
			__m256 vel_x_j = _mm256_load_ps(system->vel_x + j);
			__m256 vel_y_j = _mm256_load_ps(system->vel_y + j);
			__m256 mass_j = _mm256_load_ps(system->mass + j);

			__m256 pos_x_diff = _mm256_sub_ps(pos_x_i, pos_x_j);
			__m256 pos_y_diff = _mm256_sub_ps(pos_y_i, pos_y_j);
			__m256 distance_squared = _mm256_add_ps(_mm256_mul_ps(pos_x_diff, pos_x_diff), _mm256_mul_ps(pos_y_diff, pos_y_diff));
			__m256 distance = _mm256_min_ps(min_inv_dist_f, _mm256_rsqrt_ps(distance_squared));
			__m256 distance_cubed = _mm256_mul_ps(distance, _mm256_mul_ps(distance, distance));
			__m256 force = _mm256_mul_ps(_mm256_mul_ps(mass_i, mass_j), distance_cubed);
			__m256 force_x = _mm256_mul_ps(force, pos_x_diff);
			__m256 force_y = _mm256_mul_ps(force, pos_y_diff);

			vel_x_i = _mm256_add_ps(vel_x_i, _mm256_mul_ps(dt_f, _mm256_div_ps(force_x, mass_i)));
			vel_y_i = _mm256_add_ps(vel_y_i, _mm256_mul_ps(dt_f, _mm256_div_ps(force_y, mass_i)));

			vel_x_j = _mm256_sub_ps(vel_x_j, _mm256_mul_ps(dt_f, _mm256_div_ps(force_x, mass_j)));
			vel_y_j = _mm256_sub_ps(vel_y_j, _mm256_mul_ps(dt_f, _mm256_div_ps(force_y, mass_j)));

			_mm256_store_ps(system->vel_x + i, vel_x_i);
			_mm256_store_ps(system->vel_y + i, vel_y_i);
			_mm256_store_ps(system->vel_x + j, vel_x_j);
			_mm256_store_ps(system->vel_y + j, vel_y_j);
		}
	}

	for (size_t k = 0; k < AVX2_FLOATS; ++k) {
		for (size_t n = 0; n < MAX_PARTICLES; ++n) {
			const size_t i = n;
			const size_t j = (n + k) % MAX_PARTICLES;
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

static void attract_particles_avx512(struct particle_system *system, const float dt) {
	const float min_dist = 8*dt;
	const __m512 min_inv_dist_f = _mm512_set1_ps(1.0f / min_dist);
	const __m512 dt_f = _mm512_set1_ps(dt);

	const int use_particles = AVX512_FLOATS*(MAX_PARTICLES/AVX512_FLOATS);

	for (size_t hop = AVX512_FLOATS; hop < use_particles; ++hop) {
		for (size_t n = 0; n < use_particles/AVX512_FLOATS+1; ++n) {
			const size_t i = n*AVX512_FLOATS;
			const size_t j = (i + hop) % (use_particles);

			__m512 pos_x_i = _mm512_load_ps(system->pos_x + i);
			__m512 pos_y_i = _mm512_load_ps(system->pos_y + i);
			__m512 vel_x_i = _mm512_load_ps(system->vel_x + i);
			__m512 vel_y_i = _mm512_load_ps(system->vel_y + i);
			__m512 mass_i = _mm512_load_ps(system->mass + i);

			__m512 pos_x_j = _mm512_load_ps(system->pos_x + j);
			__m512 pos_y_j = _mm512_load_ps(system->pos_y + j);
			__m512 vel_x_j = _mm512_load_ps(system->vel_x + j);
			__m512 vel_y_j = _mm512_load_ps(system->vel_y + j);
			__m512 mass_j = _mm512_load_ps(system->mass + j);

			__m512 pos_x_diff = _mm512_sub_ps(pos_x_i, pos_x_j);
			__m512 pos_y_diff = _mm512_sub_ps(pos_y_i, pos_y_j);
			__m512 distance_squared = _mm512_add_ps(_mm512_mul_ps(pos_x_diff, pos_x_diff), _mm512_mul_ps(pos_y_diff, pos_y_diff));
			__m512 distance = _mm512_min_ps(min_inv_dist_f, _mm512_rsqrt14_ps(distance_squared));
			__m512 distance_cubed = _mm512_mul_ps(distance, _mm512_mul_ps(distance, distance));
			__m512 force = _mm512_mul_ps(_mm512_mul_ps(mass_i, mass_j), distance_cubed);
			__m512 force_x = _mm512_mul_ps(force, pos_x_diff);
			__m512 force_y = _mm512_mul_ps(force, pos_y_diff);

			vel_x_i = _mm512_add_ps(vel_x_i, _mm512_mul_ps(dt_f, _mm512_div_ps(force_x, mass_i)));
			vel_y_i = _mm512_add_ps(vel_y_i, _mm512_mul_ps(dt_f, _mm512_div_ps(force_y, mass_i)));

			vel_x_j = _mm512_sub_ps(vel_x_j, _mm512_mul_ps(dt_f, _mm512_div_ps(force_x, mass_j)));
			vel_y_j = _mm512_sub_ps(vel_y_j, _mm512_mul_ps(dt_f, _mm512_div_ps(force_y, mass_j)));

			_mm512_store_ps(system->vel_x + i, vel_x_i);
			_mm512_store_ps(system->vel_y + i, vel_y_i);
			_mm512_store_ps(system->vel_x + j, vel_x_j);
			_mm512_store_ps(system->vel_y + j, vel_y_j);
		}
	}

	for (size_t k = 0; k < AVX512_FLOATS; ++k) {
		for (size_t n = 0; n < MAX_PARTICLES; ++n) {
			const size_t i = n;
			const size_t j = (n + k) % MAX_PARTICLES;
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

static void calculate_average(struct particle_system* system, float* out_avg_x, float* out_avg_y)
{
#if USE_SIMD
#ifdef __AVX512F__
	__m512 avg_x_f = _mm512_set1_ps(0.0f);
	__m512 avg_y_f = _mm512_set1_ps(0.0f);
	for (size_t i = 0; i < MAX_PARTICLES; i += AVX512_FLOATS) {
		__m512 pos_x_f = _mm512_load_ps(system->pos_x + i);
		avg_x_f = _mm512_add_ps(avg_x_f, pos_x_f);

		__m512 pos_y_f = _mm512_load_ps(system->pos_y + i);
		avg_y_f = _mm512_add_ps(avg_y_f, pos_y_f);
	}
	(*out_avg_x) = _mm512_reduce_add_ps(avg_x_f) / (float)MAX_PARTICLES;
	(*out_avg_y) = _mm512_reduce_add_ps(avg_y_f) / (float)MAX_PARTICLES;
#elif defined __AVX2__
	__m256 avg_x_f = _mm256_set1_ps(0.0f);
	__m256 avg_y_f = _mm256_set1_ps(0.0f);
	for (size_t i = 0; i < MAX_PARTICLES; i += AVX2_FLOATS) {
		__m256 pos_x_f = _mm256_load_ps(system->pos_x + i);
		__m256 pos_y_f = _mm256_load_ps(system->pos_y + i);
		avg_x_f = _mm256_add_ps(avg_x_f, pos_x_f);
		avg_y_f = _mm256_add_ps(avg_y_f, pos_y_f);
	}
	(*out_avg_x) = hsum256_ps_avx(avg_x_f) / (float)MAX_PARTICLES;
	(*out_avg_y) = hsum256_ps_avx(avg_y_f) / (float)MAX_PARTICLES;
#else
#error SIMD not supported
#endif
#else
	float avg_x = 0.0f;
	float avg_y = 0.0f;
	for (size_t i = 0; i < MAX_PARTICLES; ++i) {
		avg_x += system->pos_x[i];
		avg_y += system->pos_y[i];
	}
	avg_x /= (float)MAX_PARTICLES;
	avg_y /= (float)MAX_PARTICLES;
	(*out_avg_x) = avg_x;
	(*out_avg_y) = avg_y;
#endif
}

static float calculate_standard_distribution(struct particle_system* system)
{
	float x_avg, y_avg;
	calculate_average(system, &x_avg, &y_avg);

#if USE_SIMD
#ifdef __AVX512F__
	const float* pos_x = system->pos_x;
	const float* pos_y = system->pos_y;

	__m512 x_dist_f = _mm512_set1_ps(0.0f);
	__m512 y_dist_f = _mm512_set1_ps(0.0f);
	const __m512 x_avg_f = _mm512_set1_ps(x_avg);
	const __m512 y_avg_f = _mm512_set1_ps(y_avg);
	for (size_t i = 0; i < MAX_PARTICLES; i += AVX512_FLOATS) {
		const __m512 pos_x_f = _mm512_load_ps(pos_x + i);
		const __m512 diff_x = _mm512_sub_ps(pos_x_f, x_avg_f);
		x_dist_f = _mm512_fmadd_ps(diff_x, diff_x, x_dist_f);

		const __m512 pos_y_f = _mm512_load_ps(pos_y + i);
		const __m512 diff_y = _mm512_sub_ps(pos_y_f, y_avg_f);
		y_dist_f = _mm512_fmadd_ps(diff_y, diff_y, y_dist_f);
	}
	const float x_dist = _mm512_reduce_add_ps(x_dist_f) / (float)MAX_PARTICLES;
	const float y_dist = _mm512_reduce_add_ps(y_dist_f) / (float)MAX_PARTICLES;
	return SDL_sqrtf(x_dist + y_dist);
#elif defined __AVX2__
	const float* pos_x = system->pos_x;
	const float* pos_y = system->pos_y;

	__m256 x_dist_f = _mm256_set1_ps(0.0f);
	__m256 y_dist_f = _mm256_set1_ps(0.0f);
	const __m256 x_avg_f = _mm256_set1_ps(x_avg);
	const __m256 y_avg_f = _mm256_set1_ps(y_avg);
	for (size_t i = 0; i < MAX_PARTICLES; i += AVX2_FLOATS) {
		const __m256 pos_x_f = _mm256_load_ps(pos_x + i);
		const __m256 diff_x = _mm256_sub_ps(pos_x_f, x_avg_f);
		x_dist_f = _mm256_fmadd_ps(diff_x, diff_x, x_dist_f);

		const __m256 pos_y_f = _mm256_load_ps(pos_y + i);
		const __m256 diff_y = _mm256_sub_ps(pos_y_f, y_avg_f);
		y_dist_f = _mm256_fmadd_ps(diff_y, diff_y, y_dist_f);
	}
	const float x_dist = hsum256_ps_avx(x_dist_f) / (float)MAX_PARTICLES;
	const float y_dist = hsum256_ps_avx(y_dist_f) / (float)MAX_PARTICLES;
	return SDL_sqrtf(x_dist + y_dist);
#else
#error SIMD not supported
#endif
#else
	float x_dist = 0.0f;
	float y_dist = 0.0f;
	for (size_t i = 0; i < MAX_PARTICLES; ++i) {
		x_dist += (system->pos_x[i] - x_avg)*(system->pos_x[i] - x_avg);
		y_dist += (system->pos_y[i] - y_avg)*(system->pos_y[i] - y_avg);
	}
	x_dist /= (float)MAX_PARTICLES;
	y_dist /= (float)MAX_PARTICLES;
	return SDL_sqrtf(x_dist + y_dist);
#endif
}

static void expand_universe(struct particle_system* system, const float amount)
{
	float avg_x, avg_y;
	calculate_average(system, &avg_x, &avg_y);

#if USE_SIMD
#ifdef __AVX512F__
	float* pos_x = system->pos_x;
	float* pos_y = system->pos_y;

	const __m512 amount_f = _mm512_set1_ps(amount);
	const __m512 avg_x_f = _mm512_set1_ps(avg_x);
	const __m512 avg_y_f = _mm512_set1_ps(avg_y);
	for (size_t i = 0; i < MAX_PARTICLES; i += AVX512_FLOATS) {
		__m512 pos_x_f = _mm512_load_ps(pos_x + i);
		pos_x_f = _mm512_mul_ps(amount_f, _mm512_sub_ps(pos_x_f, avg_x_f));
		_mm512_store_ps(pos_x + i, pos_x_f);

		__m512 pos_y_f = _mm512_load_ps(pos_y + i);
		pos_y_f = _mm512_mul_ps(amount_f, _mm512_sub_ps(pos_y_f, avg_y_f));
		_mm512_store_ps(pos_y + i, pos_y_f);
	}
#elif defined __AVX2__
	float* pos_x = system->pos_x;
	float* pos_y = system->pos_y;

	const __m256 amount_f = _mm256_set1_ps(amount);
	const __m256 avg_x_f = _mm256_set1_ps(avg_x);
	const __m256 avg_y_f = _mm256_set1_ps(avg_y);
	for (size_t i = 0; i < MAX_PARTICLES; i += AVX2_FLOATS) {
		__m256 pos_x_f = _mm256_load_ps(pos_x + i);
		pos_x_f = _mm256_mul_ps(amount_f, _mm256_sub_ps(pos_x_f, avg_x_f));
		_mm256_store_ps(pos_x + i, pos_x_f);

		__m256 pos_y_f = _mm256_load_ps(pos_y + i);
		pos_y_f = _mm256_mul_ps(amount_f, _mm256_sub_ps(pos_y_f, avg_y_f));
		_mm256_store_ps(pos_y + i, pos_y_f);
	}
#else
#error SIMD not supported
#endif

#else
	for (size_t i = 0; i < MAX_PARTICLES; ++i) {
		system->pos_x[i] = amount * (system->pos_x[i] - avg_x);
		system->pos_y[i] = amount * (system->pos_y[i] - avg_y);
	}
#endif
}

bool particle_system_update(struct particle_system* system)
{
	const float dt = 1e-6f;

	const float amount = 0.7f / calculate_standard_distribution(system);
	expand_universe(system, amount);

	const float rotation = 0.001f * SDL_powf(MAX_PARTICLES, 0.666f);
	for (size_t i = 0; i < MAX_PARTICLES; ++i) {
		system->vel_x[i] += system->pos_y[i]*rotation + ((float)pcg32_random_r(&system->rng) / (float)SDL_MAX_UINT32 * 2.0f - 1.0f);
		system->vel_y[i] += -system->pos_x[i]*rotation + ((float)pcg32_random_r(&system->rng) / (float)SDL_MAX_UINT32 * 2.0f - 1.0f);
	}

#if USE_SIMD
#ifdef __AVX512F__
	attract_particles_avx512(system, dt);
#elif defined __AVX2__
	attract_particles_avx2(system, dt);
#else
#error SIMD not supported
#endif
#else
	attract_particles(system, dt);
#endif

	move_particles(system, dt);

	copy_pos_to_points(system);

	return true;
}