#include "frame-worker.hpp"

#include <cstring>

namespace nhm {

FrameWorker::~FrameWorker() { stop(); }

bool FrameWorker::start(Job job) {
  if (thread_.joinable() || !job) return false;
  job_ = std::move(job);
  {
    std::lock_guard<std::mutex> l(mutex_);
    stop_ = false;
    has_pending_ = false;
  }
  thread_ = std::thread(&FrameWorker::run, this);
  return true;
}

void FrameWorker::stop() {
  {
    std::lock_guard<std::mutex> l(mutex_);
    stop_ = true;
  }
  cv_.notify_all();
  if (thread_.joinable()) thread_.join();
}

void FrameWorker::submit(const uint8_t *rgba, int w, int h, int linesize, float scale) {
  if (!thread_.joinable() || !rgba || w <= 0 || h <= 0 || linesize < w * 4) return;
  {
    std::lock_guard<std::mutex> l(mutex_);
    const size_t row = static_cast<size_t>(w) * 4;
    pending_.resize(row * h);
    for (int y = 0; y < h; ++y)
      std::memcpy(&pending_[row * y], rgba + static_cast<size_t>(y) * linesize, row);
    pw_ = w;
    ph_ = h;
    pscale_ = scale;
    has_pending_ = true;
  }
  cv_.notify_one();
}

void FrameWorker::run() {
  std::vector<uint8_t> buf;
  for (;;) {
    int w, h;
    float s;
    {
      std::unique_lock<std::mutex> l(mutex_);
      cv_.wait(l, [&] { return stop_ || has_pending_; });
      if (stop_) return;
      buf.swap(pending_);
      w = pw_;
      h = ph_;
      s = pscale_;
      has_pending_ = false;
    }
    try {
      job_(cv::Mat(h, w, CV_8UC4, buf.data()), s);
    } catch (...) {
      // A failed frame must never take OBS down; drop it and wait for the next.
    }
  }
}

}  // namespace nhm
