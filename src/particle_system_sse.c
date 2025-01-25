#include "particle_system_sse.h"

#include <immintrin.h>

// Inspired by the stackoverflow answer above
static float hsum_ps_sse(__m128 v) {
	__m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 1, 2, 0)); // Shuffle elements 3,1 to 2,0
	__m128 sums = _mm_add_ps(v, shuf);
	shuf        = _mm_movehl_ps(shuf, sums); // high half -> low half
	sums        = _mm_add_ss(sums, shuf);
	return        _mm_cvtss_f32(sums);
}

void attract_particles_sse(struct particle_system *system, const float dt) {
    const float min_dist = 8*dt;
	const __m128 min_inv_dist_f = _mm_set1_ps(1.0f / min_dist);
	const __m128 dt_f = _mm_set1_ps(dt);

	const size_t use_particles = SSE_FLOATS*(system->num_particles/SSE_FLOATS);

	const float* pos_x = system->pos_x;
	const float* pos_y = system->pos_y;
	float* vel_x = system->vel_x;
	float* vel_y = system->vel_y;
	const float* mass = system->mass;

	for (size_t hop = SSE_FLOATS; hop < use_particles; ++hop) {
		for (size_t n = 0; n < use_particles/SSE_FLOATS+1; ++n) {
			const size_t i = n*SSE_FLOATS;
			const size_t j = (i + hop) % (use_particles);

			const __m128 pos_x_i = _mm_loadu_ps(pos_x + i);
			const __m128 pos_y_i = _mm_loadu_ps(pos_y + i);
			__m128 vel_x_i = _mm_loadu_ps(vel_x + i);
			__m128 vel_y_i = _mm_loadu_ps(vel_y + i);
			const __m128 mass_i = _mm_loadu_ps(mass + i);

			const __m128 pos_x_j = _mm_loadu_ps(pos_x + j);
			const __m128 pos_y_j = _mm_loadu_ps(pos_y + j);
			__m128 vel_x_j = _mm_loadu_ps(vel_x + j);
			__m128 vel_y_j = _mm_loadu_ps(vel_y + j);
			const __m128 mass_j = _mm_loadu_ps(mass + j);

			const __m128 pos_x_diff = _mm_sub_ps(pos_x_i, pos_x_j);
			const __m128 pos_y_diff = _mm_sub_ps(pos_y_i, pos_y_j);
			const __m128 distance_squared = _mm_add_ps(_mm_mul_ps(pos_x_diff, pos_x_diff), _mm_mul_ps(pos_y_diff, pos_y_diff));
			const __m128 distance = _mm_min_ps(min_inv_dist_f, _mm_rsqrt_ps(distance_squared));
			const __m128 distance_cubed = _mm_mul_ps(distance, _mm_mul_ps(distance, distance));
			const __m128 force = _mm_mul_ps(_mm_mul_ps(mass_i, mass_j), distance_cubed);
			const __m128 force_x = _mm_mul_ps(force, pos_x_diff);
			const __m128 force_y = _mm_mul_ps(force, pos_y_diff);

			// If FMA is available then AVX is certainly available - use that instead.
			vel_x_i = _mm_add_ps(vel_x_i, _mm_mul_ps(dt_f, _mm_div_ps(force_x, mass_i)));
			vel_y_i = _mm_add_ps(vel_y_i, _mm_mul_ps(dt_f, _mm_div_ps(force_y, mass_i)));

			vel_x_j = _mm_sub_ps(vel_x_j, _mm_mul_ps(dt_f, _mm_div_ps(force_x, mass_j)));
			vel_y_j = _mm_sub_ps(vel_y_j, _mm_mul_ps(dt_f, _mm_div_ps(force_y, mass_j)));

			_mm_storeu_ps(vel_x + i, vel_x_i);
			_mm_storeu_ps(vel_y + i, vel_y_i);
			_mm_storeu_ps(vel_x + j, vel_x_j);
			_mm_storeu_ps(vel_y + j, vel_y_j);
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

void attract_particles_sse_batched(const struct thread_data* thread_data) {
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

	for (size_t k = thread_data->simd_range.start; k < thread_data->simd_range.end; ++k) {
		const struct particle_pair pair = pairs_simd[k];
		const size_t i = pair.i;
		const size_t j = pair.j;

		const __m128 pos_x_i = _mm_loadu_ps(pos_x + i);
		const __m128 pos_y_i = _mm_loadu_ps(pos_y + i);
		__m128 vel_x_i = _mm_loadu_ps(vel_x + i);
		__m128 vel_y_i = _mm_loadu_ps(vel_y + i);
		const __m128 mass_i = _mm_loadu_ps(mass + i);

		const __m128 pos_x_j = _mm_loadu_ps(pos_x + j);
		const __m128 pos_y_j = _mm_loadu_ps(pos_y + j);
		__m128 vel_x_j = _mm_loadu_ps(vel_x + j);
		__m128 vel_y_j = _mm_loadu_ps(vel_y + j);
		const __m128 mass_j = _mm_loadu_ps(mass + j);

		const __m128 pos_x_diff = _mm_sub_ps(pos_x_i, pos_x_j);
		const __m128 pos_y_diff = _mm_sub_ps(pos_y_i, pos_y_j);
		const __m128 distance_squared = _mm_add_ps(_mm_mul_ps(pos_x_diff, pos_x_diff), _mm_mul_ps(pos_y_diff, pos_y_diff));
		const __m128 distance = _mm_min_ps(min_inv_dist_f, _mm_rsqrt_ps(distance_squared));
		const __m128 distance_cubed = _mm_mul_ps(distance, _mm_mul_ps(distance, distance));
		const __m128 force = _mm_mul_ps(_mm_mul_ps(mass_i, mass_j), distance_cubed);
		const __m128 force_x = _mm_mul_ps(force, pos_x_diff);
		const __m128 force_y = _mm_mul_ps(force, pos_y_diff);

		// If FMA is available then AVX is certainly available - use that instead.
		vel_x_i = _mm_add_ps(vel_x_i, _mm_mul_ps(dt_f, _mm_div_ps(force_x, mass_i)));
		vel_y_i = _mm_add_ps(vel_y_i, _mm_mul_ps(dt_f, _mm_div_ps(force_y, mass_i)));

		vel_x_j = _mm_sub_ps(vel_x_j, _mm_mul_ps(dt_f, _mm_div_ps(force_x, mass_j)));
		vel_y_j = _mm_sub_ps(vel_y_j, _mm_mul_ps(dt_f, _mm_div_ps(force_y, mass_j)));

		_mm_storeu_ps(vel_x + i, vel_x_i);
		_mm_storeu_ps(vel_y + i, vel_y_i);
		_mm_storeu_ps(vel_x + j, vel_x_j);
		_mm_storeu_ps(vel_y + j, vel_y_j);
	}

	for (size_t k = thread_data->scalar_range.start; k < thread_data->scalar_range.end; ++k) {
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

void move_particles_sse(struct particle_system *system, const float dt) {
	const float* vel_x = system->vel_x;
	const float* vel_y = system->vel_y;
	float* pos_x = system->pos_x;
	float* pos_y = system->pos_y;

	const __m128 dt_f = _mm_set1_ps(dt);
	for (size_t i = 0; i < system->num_particles; i += SSE_FLOATS) {
		const __m128 vel_x_f = _mm_loadu_ps(vel_x + i);
		__m128 pos_x_f = _mm_loadu_ps(pos_x + i);
		pos_x_f = _mm_add_ps(_mm_mul_ps(dt_f, vel_x_f), pos_x_f);
		_mm_storeu_ps(pos_x + i, pos_x_f);

		const __m128 vel_y_f = _mm_loadu_ps(vel_y + i);
		__m128 pos_y_f = _mm_loadu_ps(pos_y + i);
		pos_y_f = _mm_add_ps(_mm_mul_ps(dt_f, vel_y_f), pos_y_f);
		_mm_storeu_ps(pos_y + i, pos_y_f);
	}
}

void calculate_average_sse(struct particle_system* system, float* out_avg_x, float* out_avg_y) {
	const float* pos_x = system->pos_x;
	const float* pos_y = system->pos_y;

	__m128 avg_x_f = _mm_setzero_ps();
	__m128 avg_y_f = _mm_setzero_ps();
	for (size_t i = 0; i < system->num_particles; i += SSE_FLOATS) {
		const __m128 pos_x_f = _mm_loadu_ps(pos_x + i);
		avg_x_f = _mm_add_ps(avg_x_f, pos_x_f);

		const __m128 pos_y_f = _mm_loadu_ps(pos_y + i);
		avg_y_f = _mm_add_ps(avg_y_f, pos_y_f);
	}
	(*out_avg_x) = hsum_ps_sse(avg_x_f) / (float)system->num_particles;
	(*out_avg_y) = hsum_ps_sse(avg_y_f) / (float)system->num_particles;
}

float calculate_standard_distribution_sse(struct particle_system* system) {
	float x_avg, y_avg;
	calculate_average_sse(system, &x_avg, &y_avg);

	const float* pos_x = system->pos_x;
	const float* pos_y = system->pos_y;

	__m128 x_dist_f = _mm_setzero_ps();
	__m128 y_dist_f = _mm_setzero_ps();
	const __m128 x_avg_f = _mm_set1_ps(x_avg);
	const __m128 y_avg_f = _mm_set1_ps(y_avg);
	for (size_t i = 0; i < system->num_particles; i += SSE_FLOATS) {
		const __m128 pos_x_f = _mm_loadu_ps(pos_x + i);
		const __m128 diff_x = _mm_sub_ps(pos_x_f, x_avg_f);
		x_dist_f = _mm_add_ps(_mm_mul_ps(diff_x, diff_x), x_dist_f);

		const __m128 pos_y_f = _mm_loadu_ps(pos_y + i);
		const __m128 diff_y = _mm_sub_ps(pos_y_f, y_avg_f);
		y_dist_f = _mm_add_ps(_mm_mul_ps(diff_y, diff_y), y_dist_f);
	}
	const float x_dist = hsum_ps_sse(x_dist_f) / (float)system->num_particles;
	const float y_dist = hsum_ps_sse(y_dist_f) / (float)system->num_particles;
	return SDL_sqrtf(x_dist + y_dist);
}

void expand_universe_sse(struct particle_system* system, const float amount) {
	float avg_x, avg_y;
	calculate_average_sse(system, &avg_x, &avg_y);

	float* pos_x = system->pos_x;
	float* pos_y = system->pos_y;

	const __m128 amount_f = _mm_set1_ps(amount);
	const __m128 avg_x_f = _mm_set1_ps(avg_x);
	const __m128 avg_y_f = _mm_set1_ps(avg_y);
	for (size_t i = 0; i < system->num_particles; i += SSE_FLOATS) {
		__m128 pos_x_f = _mm_loadu_ps(pos_x + i);
		pos_x_f = _mm_mul_ps(amount_f, _mm_sub_ps(pos_x_f, avg_x_f));
		_mm_storeu_ps(pos_x + i, pos_x_f);

		__m128 pos_y_f = _mm_loadu_ps(pos_y + i);
		pos_y_f = _mm_mul_ps(amount_f, _mm_sub_ps(pos_y_f, avg_y_f));
		_mm_storeu_ps(pos_y + i, pos_y_f);
	}
}

void copy_multithreaded_velocities_sse(struct particle_system* system) {
	for (size_t thread = 0; thread < system->num_threads; ++thread) {
		float* buffer_vel_x = system->buffer.vel_x[thread];
		float* buffer_vel_y = system->buffer.vel_y[thread];
		float* vel_x = system->vel_x;
		float* vel_y = system->vel_y;
		for (size_t i = 0; i < system->num_particles; i += SSE_FLOATS) {
			const __m128 buffer_vel_x_f = _mm_loadu_ps(buffer_vel_x + i);
			const __m128 buffer_vel_y_f = _mm_loadu_ps(buffer_vel_y + i);
			__m128 vel_x_f = _mm_loadu_ps(vel_x + i);
			__m128 vel_y_f = _mm_loadu_ps(vel_y + i);
			vel_x_f = _mm_add_ps(vel_x_f, buffer_vel_x_f);
			vel_y_f = _mm_add_ps(vel_y_f, buffer_vel_y_f);
			_mm_storeu_ps(vel_x + i, vel_x_f);
			_mm_storeu_ps(vel_y + i, vel_y_f);
			_mm_storeu_ps(buffer_vel_x + i, _mm_setzero_ps());
			_mm_storeu_ps(buffer_vel_y + i, _mm_setzero_ps());
		}
	}
}
