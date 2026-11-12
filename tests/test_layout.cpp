#include "check.hpp"
#include "layout.hpp"

using namespace nhm;

static MaskPose level(float yaw = 0.f, float roll = 0.f) {
  MaskPose p;
  p.cx = 500;
  p.cy = 400;
  p.scale = 180;  // 1 art px == 1 image px at size 1
  p.roll = roll;
  p.yaw = yaw;
  return p;
}

TEST(plate_unscaled_when_eye_dist_matches_art) {
  MaskLayout L = layout_mask(level(), 1.f);
  CHECK_NEAR(L.plate.p[0].x, 244, 1e-3);  // 500 - 256
  CHECK_NEAR(L.plate.p[0].y, 160, 1e-3);  // 400 - 240
  CHECK_NEAR(L.plate.p[3].x, 756, 1e-3);
  CHECK_NEAR(L.plate.p[3].y, 760, 1e-3);
  CHECK_NEAR(L.glow.p[0].x, 430, 1e-3);   // 500 - 70
  CHECK_NEAR(L.glow.p[3].y, 470, 1e-3);   // 400 + 70
}

TEST(size_doubles_plate) {
  MaskLayout L = layout_mask(level(), 2.f);
  CHECK_NEAR(L.plate.p[0].x, -12, 1e-3);  // 500 - 512
  CHECK_NEAR(L.plate.p[0].y, -80, 1e-3);  // 400 - 480
}

TEST(ears_symmetric_when_facing_camera) {
  MaskLayout L = layout_mask(level(), 1.f);
  // ear centres at (cx -/+ 1.38*180, cy + 0.45*180) = (251.6 / 748.4, 481)
  CHECK_NEAR(L.ear_l.p[0].x, 251.6 - 80, 1e-2);
  CHECK_NEAR(L.ear_l.p[0].y, 481 - 135, 1e-2);
  CHECK_NEAR(L.ear_r.p[0].x, 748.4 - 80, 1e-2);
  CHECK(!L.ear_l_behind);
  CHECK(!L.ear_r_behind);
}

TEST(roll_90_rotates_plate) {
  MaskLayout L = layout_mask(level(0.f, 1.5707963f), 1.f);
  // TL local (-256,-240) rotated +90deg -> (240,-256)
  CHECK_NEAR(L.plate.p[0].x, 740, 1e-2);
  CHECK_NEAR(L.plate.p[0].y, 144, 1e-2);
}

TEST(yaw_squashes_plate_and_hides_far_ear) {
  MaskLayout right = layout_mask(level(1.f), 1.f);
  CHECK_NEAR(right.plate.p[1].x - right.plate.p[0].x, 512 * 0.7, 1e-2);
  CHECK(right.ear_r_behind);
  CHECK(!right.ear_l_behind);
  MaskLayout left = layout_mask(level(-1.f), 1.f);
  CHECK(left.ear_l_behind);
  CHECK(!left.ear_r_behind);
}
