#include "check.hpp"
#include "mask-smoother.hpp"

using namespace nhm;

TEST(first_mask_snaps) {
  MaskSmoother s(0.5f);
  CHECK(s.empty());
  const float a[2] = {0.2f, 0.8f};
  const auto &m = s.update(a, 2, 1);
  CHECK(!s.empty());
  CHECK(s.width() == 2 && s.height() == 1);
  CHECK_NEAR(m[0], 0.2, 1e-6);
  CHECK_NEAR(m[1], 0.8, 1e-6);
}

TEST(ema_keeps_smoothing_fraction) {
  MaskSmoother s(0.5f);
  const float zero[2] = {0, 0}, one[2] = {1, 1};
  s.update(zero, 2, 1);
  const auto &m = s.update(one, 2, 1);
  CHECK_NEAR(m[0], 0.5, 1e-6);
  CHECK_NEAR(m[1], 0.5, 1e-6);
}

TEST(zero_smoothing_follows) {
  MaskSmoother s(0.f);
  const float zero[1] = {0}, one[1] = {1};
  s.update(zero, 1, 1);
  CHECK_NEAR(s.update(one, 1, 1)[0], 1.0, 1e-6);
}

TEST(smoothing_clamps_to_0_9) {
  MaskSmoother s(0.5f);
  s.set_smoothing(5.f);
  const float zero[1] = {0}, one[1] = {1};
  s.update(zero, 1, 1);
  CHECK_NEAR(s.update(one, 1, 1)[0], 0.1, 1e-6);
  s.set_smoothing(-1.f);
  CHECK_NEAR(s.update(zero, 1, 1)[0], 0.0, 1e-6);
}

TEST(size_change_snaps) {
  MaskSmoother s(0.9f);
  const float a[1] = {0};
  const float b[4] = {1, 1, 1, 1};
  s.update(a, 1, 1);
  const auto &m = s.update(b, 2, 2);
  CHECK(s.width() == 2 && s.height() == 2);
  CHECK_NEAR(m[3], 1.0, 1e-6);
}

TEST(null_or_bad_input_is_ignored) {
  MaskSmoother s(0.5f);
  const float a[1] = {0.3f};
  s.update(a, 1, 1);
  CHECK_NEAR(s.update(nullptr, 1, 1)[0], 0.3, 1e-6);
  CHECK_NEAR(s.update(a, 0, 1)[0], 0.3, 1e-6);
}

TEST(to_u8_rounds_and_clamps) {
  std::vector<uint8_t> out;
  to_u8({-0.5f, 0.f, 0.5f, 1.f, 2.f}, out);
  CHECK(out.size() == 5);
  CHECK(out[0] == 0);
  CHECK(out[1] == 0);
  CHECK(out[2] == 128);
  CHECK(out[3] == 255);
  CHECK(out[4] == 255);
}

TEST(reset_makes_next_update_snap) {
  MaskSmoother s(0.9f);
  const float zero[1] = {0}, one[1] = {1};
  s.update(zero, 1, 1);
  s.update(zero, 1, 1);
  s.reset();
  CHECK(s.empty());
  CHECK_NEAR(s.update(one, 1, 1)[0], 1.0, 1e-6);
}
