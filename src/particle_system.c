#include "particle_system.h"

#include <SDL.h>

#include "config.h"

#define SIMD_MEMORY_ALIGNMENT 32

#include <immintrin.h>

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
#elifdef __AVX2__
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