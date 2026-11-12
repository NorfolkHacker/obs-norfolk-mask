#include <chrono>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <thread>

#include "check.hpp"
#include "segmenter.hpp"

using namespace nhm;

static cv::Mat load_rgb() {
  cv::Mat bgr = cv::imread(NHM_FACE_IMG), rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  return rgb;
}

TEST(bad_model_fails_cleanly) {
  PersonSegmenter s;
  CHECK(!s.load("/nonexistent/model.onnx"));
  CHECK(!s.loaded());
  CHECK(s.segment_rgb(load_rgb()).alpha.empty());
}

TEST(segments_person_in_sample_photo) {
  PersonSegmenter s;
  CHECK(s.load(NHM_SEG_MODEL));
  SegResult r = s.segment_rgb(load_rgb());
  CHECK(r.w == 192 && r.h == 192);
  CHECK(r.alpha.size() == 192u * 192u);
  CHECK(r.seq > 0);
  CHECK(r.alpha[96 * 192 + 96] > 0.5f);  // torso near the centre
  double corner = 0;
  for (int y = 0; y < 20; ++y)
    for (int x = 0; x < 20; ++x) corner += r.alpha[y * 192 + x];
  CHECK(corner / 400.0 < 0.2);
}

TEST(async_submit_publishes) {
  PersonSegmenter s;
  CHECK(s.load(NHM_SEG_MODEL));
  cv::Mat rgba;
  cv::cvtColor(load_rgb(), rgba, cv::COLOR_RGB2RGBA);
  s.submit(rgba.data, rgba.cols, rgba.rows, static_cast<int>(rgba.step));
  SegResult r;
  for (int i = 0; i < 300 && !s.latest_if_newer(0, r); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  CHECK(r.seq > 0);
  CHECK(r.alpha.size() == 192u * 192u);
  SegResult again;
  CHECK(!s.latest_if_newer(r.seq, again));  // nothing newer yet
}
