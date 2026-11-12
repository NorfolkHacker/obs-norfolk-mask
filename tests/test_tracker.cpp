#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <thread>

#include "check.hpp"
#include "tracker.hpp"

using namespace nhm;

TEST(missing_model_fails_cleanly) {
  FaceTracker t;
  CHECK(!t.load("/nonexistent/model.onnx"));
  CHECK(!t.loaded());
}

TEST(detects_face_in_sample_photo) {
  FaceTracker t;
  CHECK(t.load(NHM_MODEL));
  cv::Mat img = cv::imread(NHM_FACE_IMG);
  CHECK(!img.empty());
  TrackResult r = t.detect_bgr(img, 1.f);
  CHECK(r.has_face);
  const Vec2 re = r.lm.pts[kRightEye], le = r.lm.pts[kLeftEye];
  CHECK(re.x < le.x);                      // right eye is on image-left
  CHECK(le.x - re.x > 6 && le.x - re.x < 25);
  CHECK(re.x > 200 && re.x < 280 && re.y > 80 && re.y < 130);
}

TEST(scale_to_src_multiplies_coords) {
  FaceTracker t;
  CHECK(t.load(NHM_MODEL));
  cv::Mat img = cv::imread(NHM_FACE_IMG);
  TrackResult a = t.detect_bgr(img, 1.f);
  TrackResult b = t.detect_bgr(img, 2.f);
  CHECK_NEAR(b.lm.pts[kNose].x, 2 * a.lm.pts[kNose].x, 1e-3);
}

TEST(async_submit_publishes_result) {
  FaceTracker t;
  CHECK(t.load(NHM_MODEL));
  cv::Mat img = cv::imread(NHM_FACE_IMG), rgba;
  cv::cvtColor(img, rgba, cv::COLOR_BGR2RGBA);
  t.submit(rgba.data, rgba.cols, rgba.rows, static_cast<int>(rgba.step), 1.f);
  TrackResult r;
  for (int i = 0; i < 200 && r.seq == 0; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    r = t.latest();
  }
  CHECK(r.seq > 0);
  CHECK(r.has_face);
}
