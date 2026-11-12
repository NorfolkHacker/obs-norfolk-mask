#pragma once
#include <cstdint>
#include <vector>

namespace nhm {

// Per-pixel exponential smoothing across successive segmentation results.
class MaskSmoother {
 public:
  explicit MaskSmoother(float smoothing = 0.5f);
  // Fraction of the previous mask kept per new result, clamped to 0..0.9.
  void set_smoothing(float smoothing);
  // Blends a new w x h result in (snaps on the first result or a size change).
  // Null or empty input leaves the current mask unchanged. Returns the current mask.
  const std::vector<float> &update(const float *fresh, int w, int h);
  // Forgets the current mask so the next update snaps.
  void reset() { cur_.clear(); w_ = h_ = 0; }
  int width() const { return w_; }
  int height() const { return h_; }
  bool empty() const { return cur_.empty(); }

 private:
  float smoothing_ = 0.5f;
  int w_ = 0, h_ = 0;
  std::vector<float> cur_;
};

// Clamps to 0..1 and converts to 0..255 with rounding.
void to_u8(const std::vector<float> &src, std::vector<uint8_t> &dst);

}  // namespace nhm
