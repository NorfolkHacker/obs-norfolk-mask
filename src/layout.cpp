#include "layout.hpp"

#include <cmath>

namespace nhm {

Quad sprite_quad(float art_w, float art_h, float anchor_x, float anchor_y, Vec2 pos, float roll,
                 float sx, float sy) {
  const float c = std::cos(roll), s = std::sin(roll);
  static const float us[4] = {0, 1, 0, 1};
  static const float vs[4] = {0, 0, 1, 1};
  Quad q;
  for (int i = 0; i < 4; ++i) {
    const float lx = (us[i] * art_w - anchor_x) * sx;
    const float ly = (vs[i] * art_h - anchor_y) * sy;
    q.p[i] = {pos.x + c * lx - s * ly, pos.y + s * lx + c * ly};
  }
  return q;
}

MaskLayout layout_mask(const MaskPose &p, float size) {
  using namespace art;
  MaskLayout L;
  const float S = p.scale * size;      // image px per inter-eye unit
  const float k = S / kPlateEyeDist;   // image px per art px
  const float ayaw = std::fabs(p.yaw);
  const float theta = p.yaw * kYawMaxRadians;
  const float c = std::cos(p.roll), s = std::sin(p.roll);
  const Vec2 centre{p.cx, p.cy};
  auto to_image = [&](float fx, float fy) {
    return Vec2{p.cx + (c * fx - s * fy) * S, p.cy + (s * fx + c * fy) * S};
  };

  const float plate_sx = k * (1.f - kYawSquash * ayaw);
  L.plate = sprite_quad(kPlateW, kPlateH, kPlateAnchorX, kPlateAnchorY, centre, p.roll, plate_sx, k);
  L.glow = sprite_quad(2 * kGlowRadius, 2 * kGlowRadius, kGlowRadius, kGlowRadius, centre, p.roll,
                       plate_sx, k);

  for (int side : {-1, +1}) {
    const float fx = side * kEarOffsetX * std::cos(theta) - kHeadDepth * std::sin(theta);
    const bool far_side = (p.yaw > kBehindYaw && side > 0) || (p.yaw < -kBehindYaw && side < 0);
    const float es = far_side ? 1.f - 0.1f * ayaw : 1.f + 0.1f * ayaw;
    const Quad q = sprite_quad(kHeadW, kHeadH, kHeadAnchorX, kHeadAnchorY,
                               to_image(fx, kEarOffsetY), p.roll, k * es, k * es);
    if (side < 0) {
      L.ear_l = q;
      L.ear_l_behind = far_side;
    } else {
      L.ear_r = q;
      L.ear_r_behind = far_side;
    }
  }
  return L;
}

}  // namespace nhm
