#include "frame-grabber.hpp"

#include <graphics/vec4.h>

namespace nhm {

gs_texture_t *FrameGrabber::capture(obs_source_t *target, uint32_t w, uint32_t h) {
  if (!target || !w || !h) return nullptr;
  if (!full_) full_ = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
  if (!full_) return nullptr;
  gs_texrender_reset(full_);
  bool ok = false;
  gs_blend_state_push();
  gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
  if (gs_texrender_begin(full_, w, h)) {
    struct vec4 clear;
    vec4_zero(&clear);
    gs_clear(GS_CLEAR_COLOR, &clear, 0.f, 0);
    gs_ortho(0.f, static_cast<float>(w), 0.f, static_cast<float>(h), -100.f, 100.f);
    obs_source_video_render(target);
    gs_texrender_end(full_);
    ok = true;
  }
  gs_blend_state_pop();
  return ok ? gs_texrender_get_texture(full_) : nullptr;
}

void FrameGrabber::stage(gs_texture_t *tex, uint32_t w, uint32_t h, uint32_t sw, uint32_t sh) {
  if (!tex || !sw || !sh) return;
  if (!small_) small_ = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
  if (!stage_ || stage_w_ != sw || stage_h_ != sh) {
    gs_stagesurface_destroy(stage_);
    stage_ = gs_stagesurface_create(sw, sh, GS_RGBA);
    stage_w_ = sw;
    stage_h_ = sh;
    pending_ = false;
  }
  if (!small_ || !stage_) return;
  gs_texrender_reset(small_);
  if (!gs_texrender_begin(small_, sw, sh)) return;
  gs_ortho(0.f, static_cast<float>(sw), 0.f, static_cast<float>(sh), -100.f, 100.f);
  gs_effect_t *def = obs_get_base_effect(OBS_EFFECT_DEFAULT);
  gs_blend_state_push();
  gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
  gs_effect_set_texture(gs_effect_get_param_by_name(def, "image"), tex);
  while (gs_effect_loop(def, "Draw")) gs_draw_sprite(tex, 0, sw, sh);
  gs_blend_state_pop();
  gs_texrender_end(small_);
  gs_stage_texture(stage_, gs_texrender_get_texture(small_));
  src_w_ = w;
  src_h_ = h;
  pending_ = true;
}

void FrameGrabber::release() {
  gs_texrender_destroy(full_);
  gs_texrender_destroy(small_);
  gs_stagesurface_destroy(stage_);
  full_ = small_ = nullptr;
  stage_ = nullptr;
  pending_ = false;
}

}  // namespace nhm
