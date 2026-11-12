#include "check.hpp"
#include "smoother.hpp"

using namespace nhm;

static MaskPose pose_at(float cx, float roll = 0.f) {
  MaskPose p;
  p.cx = cx;
  p.cy = 50;
  p.scale = 100;
  p.roll = roll;
  return p;
}

TEST(first_face_snaps_and_fades_in) {
  PoseSmoother s(0.6f);
  MaskPose p = pose_at(10);
  SmoothedPose a = s.update(&p, 0.0);
  CHECK_NEAR(a.pose.cx, 10, 1e-6);
  CHECK_NEAR(a.opacity, 0, 1e-6);
  SmoothedPose b = s.update(&p, 0.15);
  CHECK_NEAR(b.opacity, 1, 1e-5);
}

TEST(zero_smoothing_follows_immediately) {
  PoseSmoother s(0.f);
  MaskPose p = pose_at(0), q = pose_at(100);
  s.update(&p, 0.0);
  SmoothedPose r = s.update(&q, 1.0 / 30.0);
  CHECK_NEAR(r.pose.cx, 100, 1e-4);
}

TEST(one_frame_moves_by_one_minus_smoothing) {
  PoseSmoother s(0.6f);
  MaskPose p = pose_at(0), q = pose_at(100);
  s.update(&p, 0.0);
  SmoothedPose r = s.update(&q, 1.0 / 30.0);
  CHECK_NEAR(r.pose.cx, 40, 1e-3);
}

TEST(converges) {
  PoseSmoother s(0.6f);
  MaskPose p = pose_at(0), q = pose_at(100);
  s.update(&p, 0.0);
  SmoothedPose r{};
  for (int i = 1; i <= 60; ++i) r = s.update(&q, i / 30.0);
  CHECK_NEAR(r.pose.cx, 100, 0.01);
}

TEST(roll_takes_short_way_round) {
  PoseSmoother s(0.5f);
  MaskPose p = pose_at(0, 3.0f), q = pose_at(0, -2.9f);
  s.update(&p, 0.0);
  SmoothedPose r = s.update(&q, 1.0 / 30.0);
  // shortest diff is +0.3832; half of it from 3.0 = 3.1916 (wrapped)
  CHECK_NEAR(wrap_angle(r.pose.roll - 3.1916f), 0, 1e-3);
}

TEST(holds_then_fades_out) {
  PoseSmoother s(0.6f);
  MaskPose p = pose_at(10);
  s.update(&p, 0.0);
  s.update(&p, 0.15);  // opacity 1, last seen 0.15
  CHECK_NEAR(s.update(nullptr, 0.40).opacity, 1, 1e-6);  // 0.25 s since seen
  CHECK_NEAR(s.update(nullptr, 0.60).opacity, 1, 1e-6);  // 0.45 s: still holding
  SmoothedPose mid = s.update(nullptr, 0.75);             // 0.60 s: fading
  CHECK_NEAR(mid.opacity, 0.5, 1e-4);
  CHECK_NEAR(mid.pose.cx, 10, 1e-6);                      // pose held
  CHECK_NEAR(s.update(nullptr, 0.90).opacity, 0, 1e-6);
}

TEST(reacquire_after_fade_snaps) {
  PoseSmoother s(0.9f);
  MaskPose p = pose_at(10), q = pose_at(300);
  s.update(&p, 0.0);
  s.update(&p, 0.15);
  s.update(nullptr, 1.0);
  s.update(nullptr, 2.0);  // fully faded
  SmoothedPose r = s.update(&q, 2.05);
  CHECK_NEAR(r.pose.cx, 300, 1e-4);
}

TEST(invalid_pose_counts_as_no_face) {
  PoseSmoother s(0.6f);
  MaskPose p = pose_at(10), bad;  // bad.scale == 0
  s.update(&p, 0.0);
  s.update(&p, 0.15);
  s.update(&bad, 0.5);
  CHECK_NEAR(s.update(&bad, 0.95).opacity, 0, 1e-6);
}

TEST(set_smoothing_clamps) {
  PoseSmoother s(0.6f);
  s.set_smoothing(5.f);  // clamps to 0.95
  MaskPose p = pose_at(0), q = pose_at(100);
  s.update(&p, 0.0);
  SmoothedPose r = s.update(&q, 1.0 / 30.0);
  CHECK_NEAR(r.pose.cx, 5, 1e-3);
}
