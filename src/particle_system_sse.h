#pragma once

#include "particle_system.h"

void attract_particles_sse(struct particle_system *system, const float dt);
void attract_particles_sse_batched(const struct thread_data* thread_data);

void move_particles_sse(struct particle_system *system, const float dt);
void calculate_average_sse(struct particle_system* system, float* out_avg_x, float* out_avg_y);
float calculate_standard_distribution_sse(struct particle_system* system);
void expand_universe_sse(struct particle_system* system, const float amount);
void copy_multithreaded_velocities_sse(struct particle_system* system);
