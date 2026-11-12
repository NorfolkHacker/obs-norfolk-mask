// Offline check: run the tracker like the plugin does (320 px wide), lay out the
// mask with the plugin's layout code, composite it with OpenCV and save a PNG.
#include <cmath>
#include <cstdio>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

#include "layout.hpp"
#include "tracker.hpp"

using namespace nhm;

static void blit(cv::Mat &dst, const cv::Mat &bgra, const Quad &q) {
  if (bgra.empty()) return;
  const cv::Point2f src[3] = {{0, 0}, {(float)bgra.cols, 0}, {0, (float)bgra.rows}};
  const cv::Point2f d[3] = {{q.p[0].x, q.p[0].y}, {q.p[1].x, q.p[1].y}, {q.p[2].x, q.p[2].y}};
  cv::Mat warped;
  cv::warpAffine(bgra, warped, cv::getAffineTransform(src, d), dst.size(), cv::INTER_LINEAR,
                 cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0, 0));
  for (int y = 0; y < dst.rows; ++y)
    for (int x = 0; x < dst.cols; ++x) {
      const cv::Vec4b s = warped.at<cv::Vec4b>(y, x);
      const float a = s[3] / 255.f;
      if (a <= 0.f) continue;
      cv::Vec3b &o = dst.at<cv::Vec3b>(y, x);
      for (int c = 0; c < 3; ++c) o[c] = cv::saturate_cast<uchar>(o[c] * (1 - a) + s[c] * a);
    }
}

static void glow(cv::Mat &dst, const Quad &q) {
  const cv::Point2f c((q.p[0].x + q.p[3].x) / 2, (q.p[0].y + q.p[3].y) / 2);
  const float r = std::hypot(q.p[1].x - q.p[0].x, q.p[1].y - q.p[0].y) / 2;
  cv::Mat layer = cv::Mat::zeros(dst.size(), CV_8UC3);
  cv::circle(layer, c, std::max(1, (int)(r * 0.35f)), cv::Scalar(255, 38, 176), cv::FILLED, cv::LINE_AA);
  const int k = std::max(3, (int)(r * 0.6f)) | 1;
  cv::GaussianBlur(layer, layer, cv::Size(k, k), 0);
  cv::add(dst, layer, dst);
}

int main(int argc, char **argv) {
  if (argc != 5) {
    std::fprintf(stderr, "usage: %s <model.onnx> <data_dir> <in_image> <out.png>\n", argv[0]);
    return 2;
  }
  const std::string data = argv[2];
  cv::Mat img = cv::imread(argv[3]);
  if (img.empty()) {
    std::fprintf(stderr, "cannot read %s\n", argv[3]);
    return 1;
  }
  FaceTracker t;
  if (!t.load(argv[1])) {
    std::fprintf(stderr, "cannot load model %s\n", argv[1]);
    return 1;
  }
  const int sw = std::min(320, img.cols);
  const int sh = std::max(1, (int)std::lround((double)img.rows * sw / img.cols));
  cv::Mat small;
  cv::resize(img, small, cv::Size(sw, sh), 0, 0, cv::INTER_AREA);
  TrackResult r = t.detect_bgr(small, (float)img.cols / sw);
  if (!r.has_face) {
    std::fprintf(stderr, "no face found\n");
    return 1;
  }
  const MaskPose pose = compute_pose(r.lm, 0.f);
  const MaskLayout L = layout_mask(pose, 1.f);
  const cv::Mat plate = cv::imread(data + "/faceplate.png", cv::IMREAD_UNCHANGED);
  const cv::Mat el = cv::imread(data + "/headphone_l.png", cv::IMREAD_UNCHANGED);
  const cv::Mat er = cv::imread(data + "/headphone_r.png", cv::IMREAD_UNCHANGED);

  if (L.ear_l_behind) blit(img, el, L.ear_l);
  if (L.ear_r_behind) blit(img, er, L.ear_r);
  blit(img, plate, L.plate);
  if (!L.ear_l_behind) blit(img, el, L.ear_l);
  if (!L.ear_r_behind) blit(img, er, L.ear_r);
  glow(img, L.glow);
  for (const Vec2 &p : r.lm.pts) cv::circle(img, {(int)p.x, (int)p.y}, 2, cv::Scalar(0, 255, 0), cv::FILLED);

  std::printf("face score %.2f  pose cx=%.1f cy=%.1f scale=%.1f roll=%.3f yaw=%.2f\n", r.lm.score, pose.cx,
              pose.cy, pose.scale, pose.roll, pose.yaw);
  return cv::imwrite(argv[4], img) ? 0 : 1;
}
