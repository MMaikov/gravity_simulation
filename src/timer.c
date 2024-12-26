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

double timer_get_elapsed(struct timer* timer) {
    return (double)timer->sum / (double)timer->count / (double)SDL_GetPerformanceFrequency();
}