cmake_minimum_required(VERSION 3.16)
project(GravitySimulator VERSION 1.0.0)

# Set the C language standard to C11
set(CMAKE_C_STANDARD 11)

# Enable C11 features
set(CMAKE_C_STANDARD_REQUIRED ON)

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

# Automatically gather all .c files from the src directory
file(GLOB SOURCES "src/*.c")

# Automatically gather all .h files from the include directory
file(GLOB HEADERS "src/*.h")

add_executable(GravitySimulator ${SOURCES} ${HEADERS}
        src/simd_util.h
        src/simd_util.c)

# Add the SDL3 include directory, but add an alias for SDL.h
target_include_directories(GravitySimulator PRIVATE
    ${SDL3_SOURCE_DIR}/include
    ${SDL3_SOURCE_DIR}/include/SDL3  # Include SDL3 headers
    src
)

target_link_libraries(GravitySimulator PRIVATE SDL3::SDL3-static)
set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT GravitySimulator)