#include "segmenter.hpp"

#include <exception>
#include <opencv2/imgproc.hpp>

namespace nhm {

bool PersonSegmenter::load(const std::string &model_path) {
  if (loaded_) return true;
  cv::dnn::Net net;
  try {
    net = cv::dnn::readNet(model_path);
  } catch (const std::exception &) {
    return false;
  }
  if (net.empty()) return false;
  net_ = net;
  loaded_ = true;
  worker_.start([this](const cv::Mat &rgba, float) {
    cv::Mat rgb;
    cv::cvtColor(rgba, rgb, cv::COLOR_RGBA2RGB);
    SegResult r = segment_rgb(rgb);
    if (r.alpha.empty()) return;
    std::lock_guard<std::mutex> l(result_mutex_);
    result_ = std::move(r);
  });
  return true;
}

void PersonSegmenter::submit(const uint8_t *rgba, int w, int h, int linesize) {
  if (loaded_) worker_.submit(rgba, w, h, linesize, 1.f);
}

SegResult PersonSegmenter::latest() const {
  std::lock_guard<std::mutex> l(result_mutex_);
  return result_;
}

bool PersonSegmenter::latest_if_newer(uint64_t after_seq, SegResult &out) const {
  std::lock_guard<std::mutex> l(result_mutex_);
  if (result_.seq <= after_seq) return false;
  out = result_;
  return true;
}

SegResult PersonSegmenter::segment_rgb(const cv::Mat &rgb) {
  SegResult r;
  if (!loaded_ || rgb.empty() || rgb.type() != CV_8UC3) return r;
  cv::Mat in = rgb;
  if (rgb.cols != kInputSize || rgb.rows != kInputSize)
    cv::resize(rgb, in, cv::Size(kInputSize, kInputSize), 0, 0, cv::INTER_AREA);
  cv::Mat f;
  in.convertTo(f, CV_32FC3, 1.0 / 127.5, -1.0);  // (x/255 - 0.5) / 0.5
  const cv::Mat blob = cv::dnn::blobFromImage(f);  // NCHW, channels already RGB
  cv::Mat out;
  {
    std::lock_guard<std::mutex> l(net_mutex_);
    try {
      net_.setInput(blob);
      out = net_.forward().clone();
    } catch (const std::exception &) {
      out.release();
    }
  }
  if (out.dims != 4 || out.size[1] != 2 || out.size[2] != kInputSize || out.size[3] != kInputSize) return r;
  const float *person = out.ptr<float>(0, 1);  // channel 1 = person (already softmaxed)
  r.alpha.assign(person, person + kInputSize * kInputSize);
  r.w = r.h = kInputSize;
  r.seq = ++seq_;
  return r;
}

}  // namespace nhm
