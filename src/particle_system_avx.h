#pragma once

#include "particle_system.h"

void attract_particles_avx(struct particle_system *system, const float dt);
void attract_particles_avx_batched(const struct thread_data* thread_data);

void move_particles_avx(struct particle_system *system, const float dt);
void calculate_average_avx(struct particle_system* system, float* out_avg_x, float* out_avg_y);
float calculate_standard_distribution_avx(struct particle_system* system);
void expand_universe_avx(struct particle_system* system, const float amount);
void copy_multithreaded_velocities_avx(struct particle_system* system);
