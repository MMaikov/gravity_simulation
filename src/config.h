#pragma once

#define NUM_PARTICLES 1000

#define WINDOW_WIDTH 960
#define WINDOW_HEIGHT 540
#define WINDOW_TITLE "Gravity Simulator"

#define USE_SIMD 1
#define USE_FMA 1
#define USE_AVX 1
#define USE_AVX512 1

#define SSE_FLOATS 4
#define AVX_FLOATS 8
#define AVX512_FLOATS 16

// Max threads we support
#define MAX_THREADS 256

#define USE_MULTITHREADING 1