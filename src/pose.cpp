#include "pose.hpp"

#include <algorithm>
#include <cmath>

namespace nhm {

namespace {
constexpr float kPi = 3.14159265358979f;
}

float wrap_angle(float a) {
  a = std::fmod(a + kPi, 2.f * kPi);
  if (a < 0) a += 2.f * kPi;
  return a - kPi;
}

MaskPose compute_pose(const Landmarks &lm, float vertical_offset) {
  const Vec2 &re = lm.pts[kRightEye];
  const Vec2 &le = lm.pts[kLeftEye];
  const Vec2 &no = lm.pts[kNose];
  const float dx = le.x - re.x, dy = le.y - re.y;
  const float dist = std::sqrt(dx * dx + dy * dy);
  MaskPose p;
  if (dist < 1e-6f) return p;

  const float ux = dx / dist, uy = dy / dist;
  const float mx = (re.x + le.x) * 0.5f, my = (re.y + le.y) * 0.5f;
  p.roll = std::atan2(dy, dx);
  p.scale = dist;

  const float along = ((no.x - mx) * ux + (no.y - my) * uy) / dist;
  p.yaw = std::clamp(along / kYawFullOffset, -1.f, 1.f);

  // Image y points down, so the face's "down" is the eye axis rotated +90deg.
  const float downx = -uy, downy = ux;
  p.cx = mx + downx * vertical_offset * dist;
  p.cy = my + downy * vertical_offset * dist;
  return p;
}

}  // namespace nhm
