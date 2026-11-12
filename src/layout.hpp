#pragma once
#include <array>

#include "pose.hpp"

namespace nhm {

// Must match art/make_art.py.
namespace art {
constexpr float kPlateW = 512, kPlateH = 600;
constexpr float kPlateAnchorX = 256, kPlateAnchorY = 240;  // cyclops socket centre
constexpr float kPlateEyeDist = 180;                       // art px per inter-eye distance
constexpr float kHeadW = 160, kHeadH = 260;
constexpr float kHeadAnchorX = 80, kHeadAnchorY = 135;     // cup centre
constexpr float kGlowRadius = 70;                          // centred on the plate anchor
}  // namespace art

// Face-frame placement, in inter-eye units.
constexpr float kEarOffsetX = 1.38f;
constexpr float kEarOffsetY = 0.45f;
constexpr float kHeadDepth = 1.0f;       // how far ears slide opposite the nose when turning
constexpr float kYawSquash = 0.3f;       // plate width loss at |yaw| = 1
constexpr float kYawMaxRadians = 0.7854f;
constexpr float kBehindYaw = 0.15f;      // |yaw| above which the far ear goes behind the plate

// Corners in triangle-strip order TL, TR, BL, BR; UVs (0,0) (1,0) (0,1) (1,1).
struct Quad {
  std::array<Vec2, 4> p{};
};

struct MaskLayout {
  Quad plate, glow, ear_l, ear_r;  // ear_l = image-left headphone
  bool ear_l_behind = false, ear_r_behind = false;
};

// Art rect (art_w x art_h) with (anchor_x, anchor_y) placed at pos, scaled then rotated.
Quad sprite_quad(float art_w, float art_h, float anchor_x, float anchor_y, Vec2 pos, float roll,
                 float sx, float sy);

MaskLayout layout_mask(const MaskPose &pose, float size);

}  // namespace nhm
