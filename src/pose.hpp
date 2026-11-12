#pragma once
#include <array>

namespace nhm {

struct Vec2 {
  float x = 0, y = 0;
};

enum LandmarkIndex { kRightEye = 0, kLeftEye = 1, kNose = 2, kRightMouth = 3, kLeftMouth = 4 };

// YuNet's 5 points in source-pixel coordinates.
struct Landmarks {
  std::array<Vec2, 5> pts{};
  float score = 0;
};

// cx,cy: mask anchor (cyclops socket) in source px.
// scale: inter-eye distance in source px (0 = invalid).
// roll: radians, image space. yaw: -1..1, + when nose is toward the left-eye side.
struct MaskPose {
  float cx = 0, cy = 0, scale = 0, roll = 0, yaw = 0;
};

// Nose offset along the eye axis (in inter-eye units) that maps to |yaw| = 1.
constexpr float kYawFullOffset = 0.35f;

// vertical_offset is in inter-eye units along the face's "down" direction.
MaskPose compute_pose(const Landmarks &lm, float vertical_offset);

// Wrap to [-pi, pi).
float wrap_angle(float a);

}  // namespace nhm
