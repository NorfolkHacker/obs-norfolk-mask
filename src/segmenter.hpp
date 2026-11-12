#pragma once
#include <atomic>
#include <cstdint>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

#include "frame-worker.hpp"

namespace nhm {

struct SegResult {
  std::vector<float> alpha;  // w*h person probabilities, row 0 = top
  int w = 0, h = 0;
  uint64_t seq = 0;  // 0 = none yet
};

// PP-HumanSeg person segmentation (192x192, OpenCV DNN) on a background thread.
class PersonSegmenter {
 public:
  static constexpr int kInputSize = 192;

  PersonSegmenter() = default;
  PersonSegmenter(const PersonSegmenter &) = delete;
  PersonSegmenter &operator=(const PersonSegmenter &) = delete;

  // Loads the model and starts the worker. Idempotent.
  bool load(const std::string &model_path);
  bool loaded() const { return loaded_; }

  // Copies an RGBA frame of any size; the worker segments the newest one.
  void submit(const uint8_t *rgba, int w, int h, int linesize);
  SegResult latest() const;
  // Copies the latest result into out only if its seq > after_seq.
  bool latest_if_newer(uint64_t after_seq, SegResult &out) const;

  // Synchronous segmentation of an RGB (CV_8UC3) image of any size.
  SegResult segment_rgb(const cv::Mat &rgb);

 private:
  cv::dnn::Net net_;
  std::mutex net_mutex_;
  std::atomic<bool> loaded_{false};
  std::atomic<uint64_t> seq_{0};

  mutable std::mutex result_mutex_;
  SegResult result_;

  FrameWorker worker_;  // declared last: destroyed (joined) first
};

}  // namespace nhm
