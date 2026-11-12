#pragma once
#include <graphics/image-file.h>

namespace nhm {

// Loads <module data>/<module_file> as a premultiplied-alpha texture.
// Logs a warning with log_tag and returns false on failure. Call outside the
// graphics context (it enters it itself).
bool load_module_image(gs_image_file3_t *img, const char *module_file, const char *log_tag);

}  // namespace nhm
