#include "particle_system_scalar.h"

void attract_particles_scalar(struct particle_system* system, const float dt) {
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

void attract_particles_scalar_batched(const struct thread_data* thread_data) {
    const float dt = thread_data->dt;
    const float min_dist = 8*dt;

    const float* pos_x = thread_data->system->pos_x;
    const float* pos_y = thread_data->system->pos_y;
    float* vel_x = thread_data->system->buffer.vel_x[thread_data->thread_id];
    float* vel_y = thread_data->system->buffer.vel_y[thread_data->thread_id];
    const float* mass = thread_data->system->mass;

    const struct particle_pair* pairs = thread_data->system->pairs;

    for (size_t n = thread_data->scalar_range.start; n < thread_data->scalar_range.end; ++n) {
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

void move_particles_scalar(struct particle_system* system, const float dt) {
    for (size_t i = 0; i < system->num_particles; ++i) {
        system->pos_x[i] += dt * system->vel_x[i];
        system->pos_y[i] += dt * system->vel_y[i];
    }
}

void calculate_average_scalar(struct particle_system* system, float* out_avg_x, float* out_avg_y) {
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
}

float calculate_standard_distribution_scalar(struct particle_system* system) {
    float x_avg, y_avg;
    calculate_average_scalar(system, &x_avg, &y_avg);

    float x_dist = 0.0f;
    float y_dist = 0.0f;
    for (size_t i = 0; i < system->num_particles; ++i) {
        x_dist += (system->pos_x[i] - x_avg)*(system->pos_x[i] - x_avg);
        y_dist += (system->pos_y[i] - y_avg)*(system->pos_y[i] - y_avg);
    }
    x_dist /= (float)system->num_particles;
    y_dist /= (float)system->num_particles;
    return SDL_sqrtf(x_dist + y_dist);
}

void expand_universe_scalar(struct particle_system* system, const float amount) {
    float avg_x, avg_y;
    calculate_average_scalar(system, &avg_x, &avg_y);

    for (size_t i = 0; i < system->num_particles; ++i) {
        system->pos_x[i] = amount * (system->pos_x[i] - avg_x);
        system->pos_y[i] = amount * (system->pos_y[i] - avg_y);
    }
}

void copy_multithreaded_velocities_scalar(struct particle_system* system) {
    for (size_t thread = 0; thread < system->num_threads; ++thread) {
        float* buffer_vel_x = system->buffer.vel_x[thread];
        float* buffer_vel_y = system->buffer.vel_y[thread];
        float* vel_x = system->vel_x;
        float* vel_y = system->vel_y;
        for (size_t i = 0; i < system->num_particles; ++i) {
            vel_x[i] += buffer_vel_x[i];
            vel_y[i] += buffer_vel_y[i];
            buffer_vel_x[i] = 0.0f;
            buffer_vel_y[i] = 0.0f;
        }
    }
}