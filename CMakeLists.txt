cmake_minimum_required(VERSION 3.16)
project(GravitySimulator VERSION 1.0.0)

# Set the C language standard to C11
set(CMAKE_C_STANDARD 11)

# Enable C11 features
set(CMAKE_C_STANDARD_REQUIRED ON)

# Automatically gather all .h files from the include directory
file(GLOB HEADERS "src/*.h")

set(SOURCES
        src/main.c
        src/particle_system.c
        src/particle_system_scalar.c
        src/random.c
        src/simd_util.c
        src/stb_image_write.c
        src/timer.c
        src/util.c
)

try_compile(HAS_AVX512F
        ${CMAKE_BINARY_DIR}
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake_checks/cpu_avx512f.c"
        CMAKE_FLAGS -DCOMPILE_DEFINITIONS=-march=native
)

try_compile(HAS_AVX
        ${CMAKE_BINARY_DIR}
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake_checks/cpu_avx.c"
        CMAKE_FLAGS -DCOMPILE_DEFINITIONS=-march=native
)

try_compile(HAS_AVX2
        ${CMAKE_BINARY_DIR}
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake_checks/cpu_avx2.c"
        CMAKE_FLAGS -DCOMPILE_DEFINITIONS=-march=native
)

try_compile(HAS_SSE
        ${CMAKE_BINARY_DIR}
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake_checks/cpu_sse.c"
        CMAKE_FLAGS -DCOMPILE_DEFINITIONS=-march=native
)

try_compile(HAS_SSE2
        ${CMAKE_BINARY_DIR}
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake_checks/cpu_sse2.c"
        CMAKE_FLAGS -DCOMPILE_DEFINITIONS=-march=native
)

if (HAS_AVX512F)
    message(STATUS "AVX512F is supported.")
    list(APPEND SOURCES src/particle_system_avx512f.c)
endif()

if(HAS_AVX2)
    message(STATUS "AVX2 is supported.")
endif()
if(HAS_AVX)
    message(STATUS "AVX is supported.")
    list(APPEND SOURCES src/particle_system_avx.c)
endif()
if(HAS_SSE2)
    message(STATUS "SSE2 is supported.")
endif()

if(HAS_SSE)
    message(STATUS "SSE is supported.")
    list(APPEND SOURCES src/particle_system_sse.c)
endif()


if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -pedantic -march=native)
endif()

include(FetchContent)

# Fetch SDL3
FetchContent_Declare(
    SDL3
    GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
    GIT_TAG main
)

# Configure SDL3 to build as a static library
set(FETCHCONTENT_QUIET OFF)  # Optional: Show FetchContent output
set(SDL_SHARED OFF)          # Disable shared library
set(SDL_STATIC ON)           # Enable static library

# Make SDL3 available
FetchContent_MakeAvailable(SDL3)

add_executable(GravitySimulator ${SOURCES} ${HEADERS})

# Add the SDL3 include directory, but add an alias for SDL.h
target_include_directories(GravitySimulator PRIVATE
    ${SDL3_SOURCE_DIR}/include
    ${SDL3_SOURCE_DIR}/include/SDL3  # Include SDL3 headers
    src
)

target_link_libraries(GravitySimulator PRIVATE SDL3::SDL3-static)
set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT GravitySimulator)