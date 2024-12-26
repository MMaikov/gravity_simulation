#pragma once

#include <SDL.h>

struct timer
{
    uint64_t startCount;
    uint64_t endCount;

    uint64_t sum;
    uint64_t count;
};

void timer_start(struct timer* timer);
void timer_stop(struct timer* timer);
double timer_get_elapsed(struct timer* timer);