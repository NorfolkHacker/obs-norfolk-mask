#pragma once
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <opencv2/core.hpp>
#include <thread>
#include <vector>

namespace nhm {

// One background thread that always processes the newest submitted RGBA frame.
// Frames submitted while the job is busy replace each other; only the last runs.
class FrameWorker {
 public:
  // rgba is a CV_8UC4 view that is only valid during the call.
  using Job = std::function<void(const cv::Mat &rgba, float scale)>;

  FrameWorker() = default;
  ~FrameWorker();
  FrameWorker(const FrameWorker &) = delete;
  FrameWorker &operator=(const FrameWorker &) = delete;

  // Starts the thread. Returns false if already running or job is empty.
  bool start(Job job);
  bool running() const { return thread_.joinable(); }
  // Copies the frame (honouring linesize). No-op when not running.
  void submit(const uint8_t *rgba, int w, int h, int linesize, float scale);
  // Stops and joins. Safe to call repeatedly.
  void stop();

 private:
  void run();

  Job job_;
  std::thread thread_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::vector<uint8_t> pending_;
  int pw_ = 0, ph_ = 0;
  float pscale_ = 1.f;
  bool has_pending_ = false;
  bool stop_ = false;
};

}  // namespace nhm
