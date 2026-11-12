#pragma once
#include <obs-module.h>

#include <cstdint>

namespace nhm {

// Renders an OBS filter's target into a texture and stages a downscaled copy for
// the CPU; the copy is read back one frame later so the GPU never stalls.
// capture/stage/take run inside video_render; release() needs the graphics context.
class FrameGrabber {
 public:
  struct Staged {
    const uint8_t *rgba;
    uint32_t width, height, linesize;
    uint32_t src_width, src_height;  // size of the frame it was downscaled from
  };

  FrameGrabber() = default;
  FrameGrabber(const FrameGrabber &) = delete;
  FrameGrabber &operator=(const FrameGrabber &) = delete;

  // If last frame's copy is ready, maps it, calls fn(const Staged &), unmaps.
  template <class Fn>
  bool take(Fn &&fn) {
    if (!pending_ || !stage_) return false;
    pending_ = false;
    uint8_t *px = nullptr;
    uint32_t linesize = 0;
    if (!gs_stagesurface_map(stage_, &px, &linesize)) return false;
    fn(Staged{px, stage_w_, stage_h_, linesize, src_w_, src_h_});
    gs_stagesurface_unmap(stage_);
    return true;
  }

  // Discards the staged copy, e.g. after a render gap.
  void drop_pending() { pending_ = false; }

  // Renders target at w x h. Returns nullptr if rendering failed.
  gs_texture_t *capture(obs_source_t *target, uint32_t w, uint32_t h);
  // Downscales tex (w x h) to sw x sh and queues it for take() on the next frame.
  void stage(gs_texture_t *tex, uint32_t w, uint32_t h, uint32_t sw, uint32_t sh);
  void release();

 private:
  gs_texrender_t *full_ = nullptr;
  gs_texrender_t *small_ = nullptr;
  gs_stagesurf_t *stage_ = nullptr;
  uint32_t stage_w_ = 0, stage_h_ = 0, src_w_ = 0, src_h_ = 0;
  bool pending_ = false;
};

}  // namespace nhm
