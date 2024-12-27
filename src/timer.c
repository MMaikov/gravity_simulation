#include "timer.h"

void timer_start(struct timer* timer) {
    timer->startCount = SDL_GetPerformanceCounter();
}

void timer_stop(struct timer* timer) {
    timer->endCount = SDL_GetPerformanceCounter();
    const uint64_t elapsedCount = timer->endCount - timer->startCount;
    timer->sum += elapsedCount;
    timer->count += 1;
}

void timer_reset(struct timer* timer) {
    timer->sum = 0;
    timer->count = 0;
}

double timer_get_elapsed(struct timer* timer) {
    return (double)timer->sum / (double)timer->count / (double)SDL_GetPerformanceFrequency();
}

void timer_elapsed_str(struct timer* timer, size_t maxlen, char* buf) {
    double average = timer_get_elapsed(timer);
    if (average >= 1.0) {
        SDL_snprintf(buf, maxlen, "%.1f s", average);
    }
    else {
        average *= 1000.0;
        if (average >= 1.0) {
            SDL_snprintf(buf, maxlen, "%.1f ms", average);
        } else {
            average *= 1000.0;
            if (average >= 1.0) {
                SDL_snprintf(buf, maxlen, "%.1f us", average);
            } else {
                average *= 1000.0;
                SDL_snprintf(buf, maxlen, "%.1f ns", average);
            }
        }
    }
}