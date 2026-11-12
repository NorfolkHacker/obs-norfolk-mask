#include <graphics/image-file.h>
#include <graphics/vec4.h>
#include <obs-module.h>
#include <util/platform.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "frame-grabber.hpp"
#include "mask-smoother.hpp"
#include "obs-image.hpp"
#include "segmenter.hpp"

using namespace nhm;

namespace {

constexpr const char *kTag = "[norfolk-hacker-background]";
constexpr uint32_t kSeg = PersonSegmenter::kInputSize;

struct BackgroundFilter {
  obs_source_t *ctx = nullptr;
  FrameGrabber grabber;
  gs_effect_t *effect = nullptr;
  gs_texture_t *mask_tex = nullptr;
  gs_image_file3_t mill_body{}, mill_sails{};
  bool mill_ok = false;

  PersonSegmenter segmenter;
  MaskSmoother smoother;
  SegResult latest;
  std::vector<uint8_t> mask_u8;
  uint64_t last_seq = 0;

  // OBS does not render hidden sources. After a gap the staged frame, the segmenter's
  // latest result and mask_tex are all pre-gap, so drop them and show passthrough
  // until a result newer than stale_seq arrives.
  double last_render = -1;
  uint64_t stale_seq = 0;
  bool waiting_fresh = false;

  float threshold = 0.5f, feather = 0.08f, mask_smoothing = 0.5f;
  bool outline = true;
  struct vec4 outline_color {};
  float outline_width = 2.f;
  float speed = 1.f;
  bool stars = true, galaxy = true, windmill = true, grid = true, debug_mask = false;
  double anim_time = 0.0;
};

const char *bf_get_name(void *) { return "Norfolk Hacker Background"; }

void bf_update(void *data, obs_data_t *s) {
  auto *f = static_cast<BackgroundFilter *>(data);
  f->threshold = static_cast<float>(obs_data_get_double(s, "threshold"));
  f->feather = static_cast<float>(obs_data_get_double(s, "feather"));
  f->mask_smoothing = static_cast<float>(obs_data_get_double(s, "mask_smoothing"));
  f->outline = obs_data_get_bool(s, "outline");
  vec4_from_rgba(&f->outline_color, static_cast<uint32_t>(obs_data_get_int(s, "outline_color")));
  f->outline_width = static_cast<float>(obs_data_get_double(s, "outline_width"));
  f->speed = static_cast<float>(obs_data_get_double(s, "speed"));
  f->stars = obs_data_get_bool(s, "stars");
  f->galaxy = obs_data_get_bool(s, "galaxy");
  f->windmill = obs_data_get_bool(s, "windmill");
  f->grid = obs_data_get_bool(s, "grid");
  f->debug_mask = obs_data_get_bool(s, "debug_mask");
}

void bf_defaults(obs_data_t *s) {
  obs_data_set_default_double(s, "threshold", 0.5);
  obs_data_set_default_double(s, "feather", 0.08);
  obs_data_set_default_double(s, "mask_smoothing", 0.5);
  obs_data_set_default_bool(s, "outline", true);
  obs_data_set_default_int(s, "outline_color", 0xFFFF26B0);  // #B026FF (ABGR)
  obs_data_set_default_double(s, "outline_width", 2.0);
  obs_data_set_default_double(s, "speed", 1.0);
  obs_data_set_default_bool(s, "stars", true);
  obs_data_set_default_bool(s, "galaxy", true);
  obs_data_set_default_bool(s, "windmill", true);
  obs_data_set_default_bool(s, "grid", true);
  obs_data_set_default_bool(s, "debug_mask", false);
}

obs_properties_t *bf_properties(void *data) {
  auto *f = static_cast<BackgroundFilter *>(data);
  obs_properties_t *p = obs_properties_create();
  if (f && !f->segmenter.loaded())
    obs_properties_add_text(p, "status", "Segmentation model failed to load: background disabled (see OBS log)",
                            OBS_TEXT_INFO);
  obs_properties_add_float_slider(p, "threshold", "Edge threshold", 0.1, 0.9, 0.01);
  obs_properties_add_float_slider(p, "feather", "Edge softness", 0.0, 0.3, 0.01);
  obs_properties_add_float_slider(p, "mask_smoothing", "Mask smoothing", 0.0, 0.9, 0.01);
  obs_properties_add_bool(p, "outline", "Neon outline");
  obs_properties_add_color(p, "outline_color", "Outline colour");
  obs_properties_add_float_slider(p, "outline_width", "Outline width", 0.5, 6.0, 0.1);
  obs_properties_add_float_slider(p, "speed", "Animation speed", 0.0, 3.0, 0.05);
  obs_properties_add_bool(p, "stars", "Stars");
  obs_properties_add_bool(p, "galaxy", "Galaxy");
  obs_properties_add_bool(p, "windmill", "Windmill");
  obs_properties_add_bool(p, "grid", "Neon grid");
  obs_properties_add_bool(p, "debug_mask", "Debug: show cut-out");
  return p;
}

void *bf_create(obs_data_t *settings, obs_source_t *source) {
  auto *f = new BackgroundFilter();
  f->ctx = source;

  char *model = obs_module_file("models/human_segmentation_pphumanseg_2023mar.onnx");
  if (!model || !f->segmenter.load(model))
    blog(LOG_WARNING, "%s could not load segmentation model (%s); passing video through", kTag,
         model ? model : "missing");
  bfree(model);

  f->mill_ok = load_module_image(&f->mill_body, "windmill_body.png", kTag) &&
               load_module_image(&f->mill_sails, "windmill_sails.png", kTag);

  char *fx = obs_module_file("effects/background.effect");
  obs_enter_graphics();
  char *err = nullptr;
  f->effect = fx ? gs_effect_create_from_file(fx, &err) : nullptr;
  if (!f->effect) blog(LOG_WARNING, "%s failed to load effect: %s", kTag, err ? err : "file not found");
  bfree(err);
  obs_leave_graphics();
  bfree(fx);

  bf_update(f, settings);
  return f;
}

void bf_destroy(void *data) {
  auto *f = static_cast<BackgroundFilter *>(data);
  obs_enter_graphics();
  f->grabber.release();
  gs_texture_destroy(f->mask_tex);
  gs_effect_destroy(f->effect);
  gs_image_file3_free(&f->mill_body);
  gs_image_file3_free(&f->mill_sails);
  obs_leave_graphics();
  delete f;  // joins the segmenter thread
}

void bf_tick(void *data, float seconds) {
  auto *f = static_cast<BackgroundFilter *>(data);
  f->anim_time = std::fmod(f->anim_time + static_cast<double>(seconds) * f->speed, 3600.0);
}

void set_float(gs_effect_t *e, const char *name, float v) {
  gs_effect_set_float(gs_effect_get_param_by_name(e, name), v);
}

void bf_render(void *data, gs_effect_t *) {
  auto *f = static_cast<BackgroundFilter *>(data);
  obs_source_t *target = obs_filter_get_target(f->ctx);
  const uint32_t w = target ? obs_source_get_base_width(target) : 0;
  const uint32_t h = target ? obs_source_get_base_height(target) : 0;
  if (!w || !h || !f->effect || !f->segmenter.loaded()) {
    obs_source_skip_video_filter(f->ctx);
    return;
  }

  // 0. Render gap: discard everything computed before it.
  const double now = static_cast<double>(os_gettime_ns()) / 1e9;
  if (f->last_render >= 0 && now - f->last_render > 0.25) {
    f->grabber.drop_pending();
    SegResult tmp;
    f->segmenter.latest_if_newer(f->last_seq, tmp);  // skip over any pre-gap result
    f->last_seq = f->stale_seq = std::max(f->last_seq, tmp.seq);
    f->smoother.reset();  // next fresh result snaps instead of blending into the old mask
    f->waiting_fresh = true;
  }
  f->last_render = now;

  // 1. Hand last frame's 192x192 copy to the segmenter.
  f->grabber.take([&](const FrameGrabber::Staged &s) {
    f->segmenter.submit(s.rgba, static_cast<int>(s.width), static_cast<int>(s.height), static_cast<int>(s.linesize));
  });

  // 2. Capture and stage this frame.
  gs_texture_t *tex = f->grabber.capture(target, w, h);
  if (!tex) {
    obs_source_skip_video_filter(f->ctx);
    return;
  }
  f->grabber.stage(tex, w, h, kSeg, kSeg);

  // 3. New segmentation result → smooth → upload.
  if (f->segmenter.latest_if_newer(f->last_seq, f->latest)) {
    f->last_seq = f->latest.seq;
    if (f->latest.seq > f->stale_seq) f->waiting_fresh = false;
    f->smoother.set_smoothing(f->mask_smoothing);
    to_u8(f->smoother.update(f->latest.alpha.data(), f->latest.w, f->latest.h), f->mask_u8);
    if (!f->mask_tex) f->mask_tex = gs_texture_create(kSeg, kSeg, GS_R8, 1, nullptr, GS_DYNAMIC);
    if (f->mask_tex && f->mask_u8.size() == static_cast<size_t>(kSeg) * kSeg)
      gs_texture_set_image(f->mask_tex, f->mask_u8.data(), kSeg, false);
  }

  const bool prev_srgb = gs_framebuffer_srgb_enabled();
  gs_enable_framebuffer_srgb(false);
  gs_blend_state_push();
  gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);

  if (!f->mask_tex || f->waiting_fresh) {
    // No (fresh) segmentation yet: plain passthrough.
    gs_effect_t *def = obs_get_base_effect(OBS_EFFECT_DEFAULT);
    gs_effect_set_texture(gs_effect_get_param_by_name(def, "image"), tex);
    while (gs_effect_loop(def, "Draw")) gs_draw_sprite(tex, 0, w, h);
  } else {
    gs_effect_t *e = f->effect;
    gs_effect_set_texture(gs_effect_get_param_by_name(e, "image"), tex);
    gs_effect_set_texture(gs_effect_get_param_by_name(e, "mask"), f->mask_tex);
    gs_effect_set_texture(gs_effect_get_param_by_name(e, "mill_body"), f->mill_body.image2.image.texture);
    gs_effect_set_texture(gs_effect_get_param_by_name(e, "mill_sails"), f->mill_sails.image2.image.texture);
    set_float(e, "elapsed", static_cast<float>(f->anim_time));
    set_float(e, "aspect", static_cast<float>(w) / static_cast<float>(h));
    set_float(e, "threshold", f->threshold);
    set_float(e, "feather", f->feather);
    gs_effect_set_vec4(gs_effect_get_param_by_name(e, "outline_color"), &f->outline_color);
    set_float(e, "outline_on", f->outline ? 1.f : 0.f);
    set_float(e, "outline_uv", f->outline_width / static_cast<float>(kSeg));
    set_float(e, "show_stars", f->stars ? 1.f : 0.f);
    set_float(e, "show_galaxy", f->galaxy ? 1.f : 0.f);
    set_float(e, "show_mill", (f->windmill && f->mill_ok) ? 1.f : 0.f);
    set_float(e, "show_grid", f->grid ? 1.f : 0.f);
    set_float(e, "debug_mask", f->debug_mask ? 1.f : 0.f);
    while (gs_effect_loop(e, "Draw")) gs_draw_sprite(tex, 0, w, h);
  }

  gs_blend_state_pop();
  gs_enable_framebuffer_srgb(prev_srgb);
}

}  // namespace

struct obs_source_info nhm_make_background_filter_info() {
  struct obs_source_info info = {};
  info.id = "norfolk_hacker_background_filter";
  info.type = OBS_SOURCE_TYPE_FILTER;
  info.output_flags = OBS_SOURCE_VIDEO;
  info.get_name = bf_get_name;
  info.create = bf_create;
  info.destroy = bf_destroy;
  info.update = bf_update;
  info.get_defaults = bf_defaults;
  info.get_properties = bf_properties;
  info.video_tick = bf_tick;
  info.video_render = bf_render;
  return info;
}
