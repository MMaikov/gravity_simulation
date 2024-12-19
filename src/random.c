// Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)
// https://www.pcg-random.org/download.html#minimal-c-implementation

#include "random.h"

#include <stdio.h>
#include <time.h>

uint32_t pcg32_random_r(pcg32_random_t* rng)
{
    uint64_t oldstate = rng->state;
    // Advance internal state
    rng->state = oldstate * 6364136223846793005ULL + (rng->inc | 1);
    // Calculate output function (XSH RR), uses old state for max ILP
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

void pcg32_srandom_r(pcg32_random_t* rng, uint64_t initstate, uint64_t initseq)
{
    rng->state = 0U;
    rng->inc = (initseq << 1u) | 1u;
    pcg32_random_r(rng);
    rng->state += initstate;
    pcg32_random_r(rng);
}

void pcg32_srandom(pcg32_random_t* rng)
{
    pcg32_srandom_r(rng, time(NULL) ^ (intptr_t)&printf,
      (intptr_t)&puts ^ 0xDEADBEEF);
}