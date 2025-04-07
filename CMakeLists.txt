cmake_minimum_required(VERSION 3.16)
project(GravitySimulator VERSION 1.0.0)

# Set the C language standard to C11
set(CMAKE_C_STANDARD 11)

# Enable C11 features
set(CMAKE_C_STANDARD_REQUIRED ON)

# FFmpeg download URL and paths
set(FFMPEG_URL "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n7.1-latest-win64-gpl-shared-7.1.zip")
set(FFMPEG_DIR "${CMAKE_BINARY_DIR}/ffmpeg-n7.1-latest-win64-gpl-shared-7.1")
set(FFMPEG_INCLUDE_DIR "${FFMPEG_DIR}/include")
set(FFMPEG_LIBRARY_DIR "${FFMPEG_DIR}/lib")
set(FFMPEG_BIN_DIR "${FFMPEG_DIR}/bin")

set(SOURCES
        src/main.c
        src/particle_system.c
        src/particle_system_scalar.c
        src/random.c
        src/simd_util.c
        src/stb_image_write.c
        src/timer.c
        src/util.c
        src/video_encoder.c
        src/main_scalar.c
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
    list(APPEND SOURCES src/particle_system_avx512f.c src/main_avx512.c)
endif()

if(HAS_AVX2)
    message(STATUS "AVX2 is supported.")
    list(APPEND SOURCES src/main_avx2.c)
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

add_executable(GravitySimulator ${SOURCES})

# Add the SDL3 include directory, but add an alias for SDL.h
target_include_directories(GravitySimulator PRIVATE
    ${SDL3_SOURCE_DIR}/include
    ${SDL3_SOURCE_DIR}/include/SDL3  # Include SDL3 headers
    src
)

target_link_libraries(GravitySimulator SDL3::SDL3)
set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT GravitySimulator)

# Download and unzip FFmpeg if it's not already downloaded
if(WIN32)
    if(NOT EXISTS ${FFMPEG_DIR})
        message(STATUS "Downloading FFmpeg precompiled binaries...")
        file(DOWNLOAD ${FFMPEG_URL} "${CMAKE_BINARY_DIR}/ffmpeg.zip" SHOW_PROGRESS)
        message(STATUS "Unzipping FFmpeg...")
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf "${CMAKE_BINARY_DIR}/ffmpeg.zip" WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
    endif()

    # Set FFmpeg include and library directories
    if(NOT EXISTS ${FFMPEG_INCLUDE_DIR} OR NOT EXISTS ${FFMPEG_LIBRARY_DIR})
        message(FATAL_ERROR "FFmpeg was not properly extracted. Please check the download and extraction process.")
    endif()

    # Include directories
    target_include_directories(GravitySimulator PRIVATE ${FFMPEG_INCLUDE_DIR})
    target_link_directories(GravitySimulator PRIVATE ${FFMPEG_LIBRARY_DIR})
    target_link_libraries(GravitySimulator avcodec avformat avutil)

    # Copy DLL files from FFmpeg bin/ directory to the executable's directory
    file(GLOB FFMPEG_DLLS "${FFMPEG_BIN_DIR}/*.dll")
    add_custom_command(TARGET GravitySimulator POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${FFMPEG_DLLS}
            $<TARGET_FILE_DIR:GravitySimulator>)
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(LIBAV REQUIRED libavformat libavcodec libavutil)
    target_include_directories(GravitySimulator PRIVATE ${LIBAV_INCLUDE_DIRS})
    target_link_directories(GravitySimulator PRIVATE ${LIBAV_LIBRARY_DIRS})
    target_link_libraries(GravitySimulator ${LIBAV_LIBRARIES})
    add_compile_definitions(${LIBAV_CFLAGS_OTHER})
endif()