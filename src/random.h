// Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)
// https://www.pcg-random.org/download.html#minimal-c-implementation

#pragma once

#include <SDL_stdinc.h>

typedef struct { uint64_t state;  uint64_t inc; } pcg32_random_t;

#define PCG32_INITIALIZER   { 0x853c49e6748fea9bULL, 0xda3e39cb94b95bdbULL }

uint32_t pcg32_random_r(pcg32_random_t* rng);
void pcg32_srandom_r(pcg32_random_t* rng, uint64_t initstate, uint64_t initseq);
void pcg32_srandom(pcg32_random_t* rng);