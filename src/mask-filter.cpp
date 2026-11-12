#include <graphics/image-file.h>
#include <graphics/vec4.h>
#include <obs-module.h>
#include <util/platform.h>

#include <algorithm>
#include <cmath>

#include "layout.hpp"
#include "pose.hpp"
#include "smoother.hpp"
#include "tracker.hpp"

using namespace nhm;

namespace {

constexpr uint32_t kDetectWidth = 320;
constexpr const char *kTag = "[norfolk-hacker-mask]";

struct MaskFilter {
  obs_source_t *ctx = nullptr;
  gs_texrender_t *full_tr = nullptr;
  gs_texrender_t *small_tr = nullptr;
  gs_stagesurf_t *stage = nullptr;
  uint32_t stage_w = 0, stage_h = 0;
  bool stage_pending = false;
  float stage_to_src = 1.f;

  gs_effect_t *effect = nullptr;
  gs_image_file3_t plate{}, ear_l{}, ear_r{};

  FaceTracker tracker;
  PoseSmoother smoother;

  float size = 1.f;
  float v_offset = 0.f;
  float smoothing = 0.6f;
  struct vec4 glow_color {};
  float pulse_hz = 0.8f;
  bool show_headphones = true;
  bool debug_points = false;
};

void load_image(gs_image_file3_t *img, const char *name) {
  char *path = obs_module_file(name);
  gs_image_file3_init(img, path, GS_IMAGE_ALPHA_PREMULTIPLY);
  obs_enter_graphics();
  gs_image_file3_init_texture(img);
  obs_leave_graphics();
  if (!img->image2.image.loaded) blog(LOG_WARNING, "%s failed to load image %s", kTag, path ? path : name);
  bfree(path);
}

const char *mf_get_name(void *) { return "Norfolk Hacker Mask"; }

void mf_update(void *data, obs_data_t *s) {
  auto *f = static_cast<MaskFilter *>(data);
  f->size = static_cast<float>(obs_data_get_double(s, "size"));
  f->v_offset = static_cast<float>(obs_data_get_double(s, "v_offset"));
  f->smoothing = static_cast<float>(obs_data_get_double(s, "smoothing"));
  vec4_from_rgba(&f->glow_color, static_cast<uint32_t>(obs_data_get_int(s, "glow_color")));
  f->pulse_hz = static_cast<float>(obs_data_get_double(s, "pulse_hz"));
  f->show_headphones = obs_data_get_bool(s, "show_headphones");
  f->debug_points = obs_data_get_bool(s, "debug_points");
}

void mf_defaults(obs_data_t *s) {
  obs_data_set_default_double(s, "size", 1.0);
  obs_data_set_default_double(s, "v_offset", 0.0);
  obs_data_set_default_double(s, "smoothing", 0.6);
  obs_data_set_default_int(s, "glow_color", 0xFFFF26B0);  // #B026FF (OBS stores ABGR)
  obs_data_set_default_double(s, "pulse_hz", 0.8);
  obs_data_set_default_bool(s, "show_headphones", true);
  obs_data_set_default_bool(s, "debug_points", false);
}

obs_properties_t *mf_properties(void *data) {
  auto *f = static_cast<MaskFilter *>(data);
  obs_properties_t *p = obs_properties_create();
  if (f && !f->tracker.loaded())
    obs_properties_add_text(p, "status", "Face model failed to load: mask disabled (see OBS log)", OBS_TEXT_INFO);
  obs_properties_add_float_slider(p, "size", "Mask size", 0.5, 2.0, 0.01);
  obs_properties_add_float_slider(p, "v_offset", "Vertical offset", -0.5, 0.5, 0.01);
  obs_properties_add_float_slider(p, "smoothing", "Smoothing", 0.0, 0.95, 0.01);
  obs_properties_add_color(p, "glow_color", "Eye glow colour");
  obs_properties_add_float_slider(p, "pulse_hz", "Pulse speed (Hz)", 0.0, 3.0, 0.05);
  obs_properties_add_bool(p, "show_headphones", "Show headphones");
  obs_properties_add_bool(p, "debug_points", "Debug: show tracking points");
  return p;
}

void *mf_create(obs_data_t *settings, obs_source_t *source) {
  auto *f = new MaskFilter();
  f->ctx = source;

  char *model = obs_module_file("models/face_detection_yunet_2023mar.onnx");
  if (!model || !f->tracker.load(model))
    blog(LOG_WARNING, "%s could not load face model (%s); passing video through", kTag, model ? model : "missing");
  bfree(model);

  load_image(&f->plate, "faceplate.png");
  load_image(&f->ear_l, "headphone_l.png");
  load_image(&f->ear_r, "headphone_r.png");

  char *fx = obs_module_file("effects/mask.effect");
  obs_enter_graphics();
  f->full_tr = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
  f->small_tr = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
  char *err = nullptr;
  f->effect = fx ? gs_effect_create_from_file(fx, &err) : nullptr;
  if (!f->effect) blog(LOG_WARNING, "%s failed to load effect: %s", kTag, err ? err : "file not found");
  bfree(err);
  obs_leave_graphics();
  bfree(fx);

  mf_update(f, settings);
  return f;
}

void mf_destroy(void *data) {
  auto *f = static_cast<MaskFilter *>(data);
  obs_enter_graphics();
  gs_texrender_destroy(f->full_tr);
  gs_texrender_destroy(f->small_tr);
  gs_stagesurface_destroy(f->stage);
  gs_effect_destroy(f->effect);
  gs_image_file3_free(&f->plate);
  gs_image_file3_free(&f->ear_l);
  gs_image_file3_free(&f->ear_r);
  obs_leave_graphics();
  delete f;  // joins the tracker thread
}

void draw_quad(const Quad &q) {
  static const float uv[4][2] = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
  gs_render_start(true);
  for (int i = 0; i < 4; ++i) {
    gs_texcoord(uv[i][0], uv[i][1], 0);
    gs_vertex2f(q.p[i].x, q.p[i].y);
  }
  gs_render_stop(GS_TRISTRIP);
}

void draw_sprite(MaskFilter *f, gs_image_file3_t *img, const Quad &q, float opacity) {
  gs_texture_t *tex = img->image2.image.texture;
  if (!tex) return;
  gs_effect_set_texture(gs_effect_get_param_by_name(f->effect, "image"), tex);
  gs_effect_set_float(gs_effect_get_param_by_name(f->effect, "opacity"), opacity);
  while (gs_effect_loop(f->effect, "Sprite")) draw_quad(q);
}

void draw_mask(MaskFilter *f, const SmoothedPose &sp, double now) {
  const MaskLayout L = layout_mask(sp.pose, f->size);
  const enum gs_cull_mode prev_cull = gs_get_cull_mode();
  const bool prev_srgb = gs_framebuffer_srgb_enabled();
  gs_set_cull_mode(GS_NEITHER);
  gs_enable_framebuffer_srgb(false);
  gs_blend_state_push();
  gs_blend_function(GS_BLEND_ONE, GS_BLEND_INVSRCALPHA);

  if (f->show_headphones) {
    if (L.ear_l_behind) draw_sprite(f, &f->ear_l, L.ear_l, sp.opacity);
    if (L.ear_r_behind) draw_sprite(f, &f->ear_r, L.ear_r, sp.opacity);
  }
  draw_sprite(f, &f->plate, L.plate, sp.opacity);
  if (f->show_headphones) {
    if (!L.ear_l_behind) draw_sprite(f, &f->ear_l, L.ear_l, sp.opacity);
    if (!L.ear_r_behind) draw_sprite(f, &f->ear_r, L.ear_r, sp.opacity);
  }

  const float pulse =
      f->pulse_hz > 0.f ? 0.75f + 0.25f * static_cast<float>(std::sin(2.0 * M_PI * f->pulse_hz * now)) : 1.f;
  gs_blend_function(GS_BLEND_ONE, GS_BLEND_ONE);
  gs_effect_set_vec4(gs_effect_get_param_by_name(f->effect, "glow_color"), &f->glow_color);
  gs_effect_set_float(gs_effect_get_param_by_name(f->effect, "intensity"), pulse);
  gs_effect_set_float(gs_effect_get_param_by_name(f->effect, "opacity"), sp.opacity);
  while (gs_effect_loop(f->effect, "Glow")) draw_quad(L.glow);

  gs_blend_state_pop();
  gs_enable_framebuffer_srgb(prev_srgb);
  gs_set_cull_mode(prev_cull);
}

void draw_points(const Landmarks &lm) {
  gs_effect_t *solid = obs_get_base_effect(OBS_EFFECT_SOLID);
  struct vec4 c;
  vec4_set(&c, 0.f, 1.f, 0.4f, 1.f);
  gs_effect_set_vec4(gs_effect_get_param_by_name(solid, "color"), &c);
  while (gs_effect_loop(solid, "Solid"))
    for (const Vec2 &p : lm.pts) {
      Quad q;
      q.p = {Vec2{p.x - 3, p.y - 3}, Vec2{p.x + 3, p.y - 3}, Vec2{p.x - 3, p.y + 3}, Vec2{p.x + 3, p.y + 3}};
      draw_quad(q);
    }
}

void mf_render(void *data, gs_effect_t *) {
  auto *f = static_cast<MaskFilter *>(data);
  obs_source_t *target = obs_filter_get_target(f->ctx);
  const uint32_t w = target ? obs_source_get_base_width(target) : 0;
  const uint32_t h = target ? obs_source_get_base_height(target) : 0;
  if (!w || !h || !f->effect) {
    obs_source_skip_video_filter(f->ctx);
    return;
  }

  // 1. Hand last frame's staged pixels to the tracker (mapping a frame late avoids a GPU stall).
  if (f->stage_pending) {
    uint8_t *px = nullptr;
    uint32_t linesize = 0;
    if (gs_stagesurface_map(f->stage, &px, &linesize)) {
      f->tracker.submit(px, static_cast<int>(f->stage_w), static_cast<int>(f->stage_h), static_cast<int>(linesize),
                        f->stage_to_src);
      gs_stagesurface_unmap(f->stage);
    }
    f->stage_pending = false;
  }

  // 2. Capture the source at full size.
  gs_texrender_reset(f->full_tr);
  gs_blend_state_push();
  gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
  if (gs_texrender_begin(f->full_tr, w, h)) {
    struct vec4 clear;
    vec4_zero(&clear);
    gs_clear(GS_CLEAR_COLOR, &clear, 0.f, 0);
    gs_ortho(0.f, static_cast<float>(w), 0.f, static_cast<float>(h), -100.f, 100.f);
    obs_source_video_render(target);
    gs_texrender_end(f->full_tr);
  }
  gs_blend_state_pop();
  gs_texture_t *tex = gs_texrender_get_texture(f->full_tr);
  if (!tex) {
    obs_source_skip_video_filter(f->ctx);
    return;
  }

  // 3. Downscale and stage for the tracker.
  gs_effect_t *def = obs_get_base_effect(OBS_EFFECT_DEFAULT);
  if (f->tracker.loaded()) {
    const uint32_t sw = std::min(kDetectWidth, w);
    const uint32_t sh = std::max(1u, static_cast<uint32_t>(std::lround(static_cast<double>(h) * sw / w)));
    if (!f->stage || f->stage_w != sw || f->stage_h != sh) {
      gs_stagesurface_destroy(f->stage);
      f->stage = gs_stagesurface_create(sw, sh, GS_RGBA);
      f->stage_w = sw;
      f->stage_h = sh;
    }
    gs_texrender_reset(f->small_tr);
    if (f->stage && gs_texrender_begin(f->small_tr, sw, sh)) {
      gs_ortho(0.f, static_cast<float>(sw), 0.f, static_cast<float>(sh), -100.f, 100.f);
      gs_blend_state_push();
      gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
      gs_effect_set_texture(gs_effect_get_param_by_name(def, "image"), tex);
      while (gs_effect_loop(def, "Draw")) gs_draw_sprite(tex, 0, sw, sh);
      gs_blend_state_pop();
      gs_texrender_end(f->small_tr);
      gs_stage_texture(f->stage, gs_texrender_get_texture(f->small_tr));
      f->stage_to_src = static_cast<float>(w) / static_cast<float>(sw);
      f->stage_pending = true;
    }
  }

  // 4. Draw the camera image.
  const bool prev_srgb = gs_framebuffer_srgb_enabled();
  gs_enable_framebuffer_srgb(true);
  gs_effect_set_texture_srgb(gs_effect_get_param_by_name(def, "image"), tex);
  while (gs_effect_loop(def, "Draw")) gs_draw_sprite(tex, 0, w, h);
  gs_enable_framebuffer_srgb(prev_srgb);

  // 5. Pose → smoothing → mask.
  const TrackResult r = f->tracker.latest();
  MaskPose pose;
  const MaskPose *target_pose = nullptr;
  if (r.has_face) {
    pose = compute_pose(r.lm, f->v_offset);
    if (pose.scale > 0) target_pose = &pose;
  }
  f->smoother.set_smoothing(f->smoothing);
  const double now = static_cast<double>(os_gettime_ns()) / 1e9;
  const SmoothedPose sp = f->smoother.update(target_pose, now);
  if (sp.opacity > 0.f) draw_mask(f, sp, now);
  if (f->debug_points && r.has_face) draw_points(r.lm);
}

}  // namespace

struct obs_source_info nhm_make_mask_filter_info() {
  struct obs_source_info info = {};
  info.id = "norfolk_hacker_mask_filter";
  info.type = OBS_SOURCE_TYPE_FILTER;
  info.output_flags = OBS_SOURCE_VIDEO;
  info.get_name = mf_get_name;
  info.create = mf_create;
  info.destroy = mf_destroy;
  info.update = mf_update;
  info.get_defaults = mf_defaults;
  info.get_properties = mf_properties;
  info.video_render = mf_render;
  return info;
}
