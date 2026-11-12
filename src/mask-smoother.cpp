#include "mask-smoother.hpp"

#include <algorithm>
#include <cmath>

namespace nhm {

MaskSmoother::MaskSmoother(float smoothing) { set_smoothing(smoothing); }

void MaskSmoother::set_smoothing(float smoothing) { smoothing_ = std::clamp(smoothing, 0.f, 0.9f); }

const std::vector<float> &MaskSmoother::update(const float *fresh, int w, int h) {
  if (!fresh || w <= 0 || h <= 0) return cur_;
  const size_t n = static_cast<size_t>(w) * h;
  if (w != w_ || h != h_ || cur_.size() != n) {
    cur_.assign(fresh, fresh + n);
    w_ = w;
    h_ = h;
    return cur_;
  }
  const float a = 1.f - smoothing_;
  for (size_t i = 0; i < n; ++i) cur_[i] += (fresh[i] - cur_[i]) * a;
  return cur_;
}

void to_u8(const std::vector<float> &src, std::vector<uint8_t> &dst) {
  dst.resize(src.size());
  for (size_t i = 0; i < src.size(); ++i)
    dst[i] = static_cast<uint8_t>(std::lround(std::clamp(src[i], 0.f, 1.f) * 255.f));
}

}  // namespace nhm
