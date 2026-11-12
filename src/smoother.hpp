#pragma once
#include "pose.hpp"

namespace nhm {

struct SmoothedPose {
  MaskPose pose;
  float opacity = 0;
};

class PoseSmoother {
 public:
  static constexpr double kHoldSeconds = 0.5;
  static constexpr double kFadeOutSeconds = 0.3;
  static constexpr double kFadeInSeconds = 0.15;

  explicit PoseSmoother(float smoothing = 0.6f);
  // 0 = raw, 0.95 = very smooth. Fraction of the old value kept per 1/30 s.
  void set_smoothing(float smoothing);
  // target == nullptr or target->scale <= 0 means no face this frame.
  SmoothedPose update(const MaskPose *target, double now_seconds);

 private:
  float smoothing_;
  bool tracking_ = false;  // true from first sight until a fade-out completes
  MaskPose cur_{};
  float opacity_ = 0;
  double last_time_ = -1;
  double last_seen_ = -1;
};

}  // namespace nhm
