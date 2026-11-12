#include <obs-module.h>

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE("norfolk-hacker-mask", "en-US")

struct obs_source_info nhm_make_mask_filter_info();

bool obs_module_load(void) {
  struct obs_source_info info = nhm_make_mask_filter_info();
  obs_register_source(&info);
  blog(LOG_INFO, "[norfolk-hacker-mask] loaded version %s", "0.1.0");
  return true;
}
