#include "check.hpp"
#include "pose.hpp"

using namespace nhm;

static Landmarks make(Vec2 re, Vec2 le, Vec2 nose) {
  Landmarks lm;
  lm.pts[kRightEye] = re;
  lm.pts[kLeftEye] = le;
  lm.pts[kNose] = nose;
  lm.pts[kRightMouth] = {re.x, re.y + 100};
  lm.pts[kLeftMouth] = {le.x, le.y + 100};
  lm.score = 0.9f;
  return lm;
}

TEST(level_face) {
  MaskPose p = compute_pose(make({100, 200}, {200, 200}, {150, 250}), 0.f);
  CHECK_NEAR(p.cx, 150, 1e-4);
  CHECK_NEAR(p.cy, 200, 1e-4);
  CHECK_NEAR(p.scale, 100, 1e-4);
  CHECK_NEAR(p.roll, 0, 1e-6);
  CHECK_NEAR(p.yaw, 0, 1e-6);
}

TEST(tilted_face_roll) {
  MaskPose p = compute_pose(make({100, 100}, {200, 200}, {140, 160}), 0.f);
  CHECK_NEAR(p.roll, 0.7853982, 1e-5);
  CHECK_NEAR(p.scale, 141.42136, 1e-3);
}

TEST(yaw_from_nose_offset) {
  CHECK_NEAR(compute_pose(make({100, 200}, {200, 200}, {167.5f, 250}), 0.f).yaw, 0.5, 1e-4);
  CHECK_NEAR(compute_pose(make({100, 200}, {200, 200}, {132.5f, 250}), 0.f).yaw, -0.5, 1e-4);
  CHECK_NEAR(compute_pose(make({100, 200}, {200, 200}, {250, 250}), 0.f).yaw, 1.0, 1e-6);
  CHECK_NEAR(compute_pose(make({100, 200}, {200, 200}, {0, 250}), 0.f).yaw, -1.0, 1e-6);
}

TEST(vertical_offset_follows_face_down) {
  MaskPose level = compute_pose(make({100, 200}, {200, 200}, {150, 250}), 0.5f);
  CHECK_NEAR(level.cx, 150, 1e-4);
  CHECK_NEAR(level.cy, 250, 1e-4);
  // eyes stacked vertically: roll 90deg, face "down" points to -x
  MaskPose side = compute_pose(make({100, 100}, {100, 200}, {80, 150}), 0.5f);
  CHECK_NEAR(side.cx, 50, 1e-3);
  CHECK_NEAR(side.cy, 150, 1e-3);
}

TEST(degenerate_eyes_invalid) {
  MaskPose p = compute_pose(make({100, 100}, {100, 100}, {100, 120}), 0.f);
  CHECK(p.scale == 0.f);
}

TEST(wrap_angle_range) {
  CHECK_NEAR(wrap_angle(3 * 3.14159265f / 2), -3.14159265 / 2, 1e-5);
  CHECK_NEAR(wrap_angle(-3 * 3.14159265f / 2), 3.14159265 / 2, 1e-5);
  CHECK_NEAR(wrap_angle(0.5f), 0.5, 1e-6);
}
