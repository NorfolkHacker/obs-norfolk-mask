#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#include <string>
#include <thread>
#include <vector>

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
  ~FaceTracker();
  FaceTracker(const FaceTracker &) = delete;
  FaceTracker &operator=(const FaceTracker &) = delete;

  bool load(const std::string &model_path);
  bool loaded() const { return loaded_; }

  // Copies the frame; the worker thread detects on the newest submitted frame.
  void submit(const uint8_t *rgba, int w, int h, int linesize, float scale_to_src);
  TrackResult latest() const;

  // Synchronous detection (also used by the worker).
  TrackResult detect_bgr(const cv::Mat &bgr, float scale_to_src);

 private:
  void run();

  cv::Ptr<cv::FaceDetectorYN> det_;
  std::mutex det_mutex_;
  bool loaded_ = false;
  std::atomic<uint64_t> seq_{0};

  std::thread worker_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::vector<uint8_t> pending_;
  int pw_ = 0, ph_ = 0;
  float pscale_ = 1.f;
  bool has_pending_ = false;
  bool stop_ = false;

  mutable std::mutex result_mutex_;
  TrackResult result_;
};

}  // namespace nhm
