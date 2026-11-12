#include "tracker.hpp"

#include <exception>
#include <opencv2/imgproc.hpp>

namespace nhm {

bool FaceTracker::load(const std::string &model_path) {
  if (loaded_) return true;
  cv::Ptr<cv::FaceDetectorYN> det;
  try {
    det = cv::FaceDetectorYN::create(model_path, "", cv::Size(320, 320), kScoreThreshold, 0.3f, 5);
  } catch (const std::exception &) {
    det.release();
  }
  if (det.empty()) return false;
  det_ = det;
  loaded_ = true;
  worker_.start([this](const cv::Mat &rgba, float scale) {
    cv::Mat bgr;
    cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
    TrackResult r = detect_bgr(bgr, scale);
    std::lock_guard<std::mutex> l(result_mutex_);
    result_ = r;
  });
  return true;
}

void FaceTracker::submit(const uint8_t *rgba, int w, int h, int linesize, float scale_to_src) {
  if (loaded_) worker_.submit(rgba, w, h, linesize, scale_to_src);
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
    } catch (const std::exception &) {
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

}  // namespace nhm
