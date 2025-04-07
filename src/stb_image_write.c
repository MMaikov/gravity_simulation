#include <SDL.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBIW_ASSERT(x) SDL_assert(x)
#define STBIW_MALLOC(x) SDL_malloc(x)
#define STBIW_REALLOC(x, y) SDL_realloc(x, y)
#define STBIW_FREE(x) SDL_free(x)
#define STBIW_MEMMOVE(x, y, z) SDL_memmove(x, y, z)
#define STBI_WRITE_NO_STDIO
#include "stb_image_write.h"