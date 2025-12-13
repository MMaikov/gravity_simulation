#pragma once

#define NUM_PARTICLES 1000

#define WINDOW_WIDTH 960
#define WINDOW_HEIGHT 540
#define WINDOW_TITLE "Gravity Simulator"

#define VIDEO_OUTPUT 1
#define RECORDING_FPS 30

#define USE_SIMD 1
#define USE_AVX 1
#define USE_AVX512 1

#define SSE_FLOATS 4
#define AVX_FLOATS 8
#define AVX512_FLOATS 16

// Max threads we support
#define MAX_THREADS 256

#define USE_MULTITHREADING 1

#define PARTICLE_SYSTEM_ADD_RANDOM_FORCE 1

#define AVX512_AVOID_GATHERSCATTER 0

// from Google's Chromium project
#define COUNT_OF(x) ((sizeof(x)/sizeof(0[x])) / ((size_t)(!(sizeof(x) % sizeof(0[x]))))) //NOLINT