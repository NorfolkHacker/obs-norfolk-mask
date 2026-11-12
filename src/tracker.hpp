#pragma once
#include <atomic>
#include <cstdint>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#include <string>

#include "frame-worker.hpp"
#include "pose.hpp"

namespace nhm {

struct TrackResult {
  bool has_face = false;
  Landmarks lm{};
  uint64_t seq = 0;  // increments per completed detection; 0 = none yet
};

class FaceTracker {
 public:
  static constexpr float kScoreThreshold = 0.6f;

  FaceTracker() = default;
  FaceTracker(const FaceTracker &) = delete;
  FaceTracker &operator=(const FaceTracker &) = delete;

  // Loads the model and starts the worker. Idempotent: later calls just return loaded().
  bool load(const std::string &model_path);
  bool loaded() const { return loaded_; }

  // Copies the frame; the worker detects on the newest submitted frame.
  void submit(const uint8_t *rgba, int w, int h, int linesize, float scale_to_src);
  TrackResult latest() const;

  // Synchronous detection (also used by the worker).
  TrackResult detect_bgr(const cv::Mat &bgr, float scale_to_src);

 private:
  cv::Ptr<cv::FaceDetectorYN> det_;
  std::mutex det_mutex_;
  std::atomic<bool> loaded_{false};
  std::atomic<uint64_t> seq_{0};

  mutable std::mutex result_mutex_;
  TrackResult result_;

  FrameWorker worker_;  // declared last: destroyed (joined) first
};

}  // namespace nhm
