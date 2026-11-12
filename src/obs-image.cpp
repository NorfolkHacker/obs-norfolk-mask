#include "obs-image.hpp"

#include <obs-module.h>

namespace nhm {

bool load_module_image(gs_image_file3_t *img, const char *module_file, const char *log_tag) {
  char *path = obs_module_file(module_file);
  gs_image_file3_init(img, path, GS_IMAGE_ALPHA_PREMULTIPLY);
  obs_enter_graphics();
  gs_image_file3_init_texture(img);
  obs_leave_graphics();
  const bool ok = img->image2.image.loaded && img->image2.image.texture;
  if (!ok) blog(LOG_WARNING, "%s failed to load image %s", log_tag, path ? path : module_file);
  bfree(path);
  return ok;
}

}  // namespace nhm
