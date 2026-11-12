#include "tracker.hpp"

#include <cstring>
#include <opencv2/imgproc.hpp>

namespace nhm {

FaceTracker::~FaceTracker() {
  {
    std::lock_guard<std::mutex> l(mutex_);
    stop_ = true;
  }
  cv_.notify_all();
  if (worker_.joinable()) worker_.join();
}

bool FaceTracker::load(const std::string &model_path) {
  try {
    det_ = cv::FaceDetectorYN::create(model_path, "", cv::Size(320, 320), kScoreThreshold, 0.3f, 5);
  } catch (const cv::Exception &) {
    det_.release();
  }
  loaded_ = !det_.empty();
  if (loaded_ && !worker_.joinable()) worker_ = std::thread(&FaceTracker::run, this);
  return loaded_;
}

void FaceTracker::submit(const uint8_t *rgba, int w, int h, int linesize, float scale_to_src) {
  if (!loaded_ || !rgba || w <= 0 || h <= 0) return;
  {
    std::lock_guard<std::mutex> l(mutex_);
    pending_.resize(static_cast<size_t>(w) * h * 4);
    for (int y = 0; y < h; ++y)
      std::memcpy(&pending_[static_cast<size_t>(y) * w * 4], rgba + static_cast<size_t>(y) * linesize,
                  static_cast<size_t>(w) * 4);
    pw_ = w;
    ph_ = h;
    pscale_ = scale_to_src;
    has_pending_ = true;
  }
  cv_.notify_one();
}

TrackResult FaceTracker::latest() const {
  std::lock_guard<std::mutex> l(result_mutex_);
  return result_;
}

TrackResult FaceTracker::detect_bgr(const cv::Mat &bgr, float scale_to_src) {
  TrackResult r;
  if (!loaded_ || bgr.empty()) return r;
  cv::Mat faces;
  {
    std::lock_guard<std::mutex> l(det_mutex_);
    try {
      det_->setInputSize(bgr.size());
      det_->detect(bgr, faces);
    } catch (const cv::Exception &) {
      faces.release();
    }
  }
  int best = -1;
  float best_score = kScoreThreshold;
  for (int i = 0; i < faces.rows; ++i) {
    const float sc = faces.at<float>(i, 14);
    if (sc >= best_score) {
      best_score = sc;
      best = i;
    }
  }
  if (best >= 0) {
    r.has_face = true;
    for (int k = 0; k < 5; ++k)
      r.lm.pts[k] = {faces.at<float>(best, 4 + 2 * k) * scale_to_src,
                     faces.at<float>(best, 5 + 2 * k) * scale_to_src};
    r.lm.score = best_score;
  }
  r.seq = ++seq_;
  return r;
}

void FaceTracker::run() {
  std::vector<uint8_t> buf;
  cv::Mat bgr;
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
    cv::Mat rgba(h, w, CV_8UC4, buf.data());
    cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
    TrackResult r = detect_bgr(bgr, s);
    std::lock_guard<std::mutex> l(result_mutex_);
    result_ = r;
  }
}

}  // namespace nhm
