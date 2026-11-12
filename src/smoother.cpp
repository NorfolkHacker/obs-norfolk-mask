#include "smoother.hpp"

#include <algorithm>
#include <cmath>

namespace nhm {

PoseSmoother::PoseSmoother(float smoothing) { set_smoothing(smoothing); }

void PoseSmoother::set_smoothing(float smoothing) {
  smoothing_ = std::clamp(smoothing, 0.f, 0.95f);
}

SmoothedPose PoseSmoother::update(const MaskPose *target, double now) {
  const double dt = last_time_ < 0 ? 0.0 : std::max(0.0, now - last_time_);
  last_time_ = now;

  if (target && target->scale > 0) {
    if (!tracking_) {
      cur_ = *target;  // snap on first sight / after a full fade
      tracking_ = true;
    } else {
      const float keep = smoothing_ <= 0.f ? 0.f : static_cast<float>(std::pow(smoothing_, dt * 30.0));
      const float a = 1.f - keep;
      cur_.cx += (target->cx - cur_.cx) * a;
      cur_.cy += (target->cy - cur_.cy) * a;
      cur_.scale += (target->scale - cur_.scale) * a;
      cur_.yaw += (target->yaw - cur_.yaw) * a;
      cur_.roll = wrap_angle(cur_.roll + wrap_angle(target->roll - cur_.roll) * a);
    }
    last_seen_ = now;
    opacity_ = std::min(1.f, opacity_ + static_cast<float>(dt / kFadeInSeconds));
  } else if (tracking_ && now - last_seen_ > kHoldSeconds) {
    opacity_ = std::max(0.f, opacity_ - static_cast<float>(dt / kFadeOutSeconds));
    if (opacity_ <= 0.f) tracking_ = false;
  }
  return {cur_, opacity_};
}

}  // namespace nhm
